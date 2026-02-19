# Notion에 바로 붙여쓰기 용 (단일 블록)

import os
import re
import numpy as np
import pandas as pd
import open3d as o3d
from .utils.fieldscale import Fieldscale
from .base import BaseDataset
import matplotlib.pyplot as plt

S_cv_from_ned  = np.array([[0,1,0],
                           [0,0,1],
                           [1,0,0]], dtype=float)
S_ned_from_cv  = S_cv_from_ned.T  # inverse

# Reverse mapping: raw dataset name → friendly name (for SLAM pose lookup)
_WHEEL_FRIENDLY_NAMES = {
    '25_10_20_14_50': 'indoor',
    '25_10_20_14_30': 'outdoor',
    '25_11_04_16_00': 'forest',
}

# Sub-dataset frame ranges (for split datasets)
_WHEEL_SUB_DATASETS = {
    'forest_1': ('25_11_04_16_00', 1, 2369),
    'forest_2': ('25_11_04_16_00', 2371, 2860),
    'forest_3': ('25_11_04_16_00', 2862, 7405),
}

# ----------------------------
# 유틸
# ----------------------------
def _load_frame_ts(npy_path: str) -> float:
    a = np.asarray(np.load(npy_path))
    return float(a if a.ndim == 0 else a.reshape(-1)[0])

def _load_odom_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 시간 열 파싱
    if 'stamp' in df.columns:
        t = df['stamp'].to_numpy(dtype=np.float64)          # 초(float) 가정
    elif {'sec','nsec'}.issubset(df.columns):
        t = df['sec'].to_numpy(dtype=np.int64) + df['nsec'].to_numpy(dtype=np.int64)*1e-9
    else:
        raise ValueError("odom csv에는 'stamp' 또는 ['sec','nsec']가 필요합니다.")

    order = np.argsort(t)
    df = df.iloc[order].reset_index(drop=True).copy()
    df['stamp_sec'] = t[order]
    return df

def _infer_linear_speed(odom_df: pd.DataFrame) -> np.ndarray:
    """
    선속도 [m/s] 추정. 컬럼 상황에 따라 자동 선택:
    - 'linear_x' 있으면 그걸 사용(부호 무시, 절댓값)
    - 'vx','vy' 두 축 있으면 노름 사용
    - 'speed' 있으면 사용
    - 위가 없고 'pos_x','pos_y'만 있으면 위치 미분으로 근사
    """
    cols = odom_df.columns
    if 'linear_x' in cols:
        v = np.abs(odom_df['linear_x'].to_numpy(dtype=np.float64))
        return v
    if {'vx','vy'}.issubset(cols):
        vx = odom_df['vx'].to_numpy(dtype=np.float64)
        vy = odom_df['vy'].to_numpy(dtype=np.float64)
        return np.sqrt(vx*vx + vy*vy)
    if 'speed' in cols:
        return np.abs(odom_df['speed'].to_numpy(dtype=np.float64))

    # 위치 미분으로 근사 (방향은 신뢰 안 하더라도 속도 크기는 얻기 위함)
    if {'pos_x','pos_y'}.issubset(cols):
        t  = odom_df['stamp_sec'].to_numpy(dtype=np.float64)
        x  = odom_df['pos_x'].to_numpy(dtype=np.float64)
        y  = odom_df['pos_y'].to_numpy(dtype=np.float64)
        dt = np.clip(np.diff(t), 1e-6, None)
        ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        v_mid = ds / dt
        # 양끝 보정: 길이를 맞추기 위해 양 끝을 복제
        v = np.empty_like(t, dtype=np.float64)
        v[0]  = v_mid[0]
        v[-1] = v_mid[-1]
        if v.size > 2:
            v[1:-1] = 0.5*(v_mid[:-1] + v_mid[1:])
        return v

    raise ValueError("선속도를 추정할 수 있는 컬럼이 없습니다. (linear_x | vx,vy | speed | pos_x,pos_y 중 하나 필요)")

def _cum_arc_length(t_odom: np.ndarray, v_lin: np.ndarray) -> np.ndarray:
    """
    선속도 적분 → 누적거리 S(t) [m]. 사다리꼴 적분.
    """
    t = np.asarray(t_odom, dtype=np.float64)
    v = np.asarray(v_lin,  dtype=np.float64)
    if t.size < 2:
        return np.zeros_like(t)
    dt   = np.diff(t)
    midv = 0.5*(v[1:]+v[:-1])
    incr = dt*midv
    S = np.concatenate(([0.0], np.cumsum(incr)))
    return S

# ----------------------------
# 메인 로더
# ----------------------------
class WheelLoader(BaseDataset):
    """
    변경 사항 핵심:
    - 오돔에서 선속도 -> 누적거리 S(t) 적분
    - 카메라 프레임 시각으로 S를 보간하여 사용
    - 방향은 무시하므로, get_odom(idx)에서 x=S, y=0을 반환 (직진 가정)
      (호→현 보정은 외부 모듈에서 수행)
    """
    def __init__(self, name="25_10_20_14_50", data_type="rgb"):
        # Support sub-dataset names (e.g. 'forest_1' -> '25_11_04_16_00' with frame range)
        self._frame_range = None
        if name in _WHEEL_SUB_DATASETS:
            raw_name, start_frame, end_frame = _WHEEL_SUB_DATASETS[name]
            self._friendly_override = name
            self._frame_range = (start_frame, end_frame)
            name = raw_name
        else:
            self._friendly_override = None

        self.data_type      = data_type
        self.dataset_path   = f"/home/nas/Dataset3/SNU_data/{name}"
        self.name           = name
        self.timestamp_path = os.path.join(self.dataset_path, "timestamp")
        self.depth_path     = os.path.join(self.dataset_path, "LiDAR_map")
        self.lidar_path     = os.path.join(self.dataset_path, "LiDAR")
        self.rgb_path       = os.path.join(self.dataset_path, "top")
        self.thr_path       = os.path.join(self.dataset_path, "thermal")
        self.odom_csv       = os.path.join(self.dataset_path, "cmd", "odom_log.csv")  # 필요 시 경로 수정

        self.fx=408.90311623; self.fy=408.66620522
        self.cx=309.32120284; self.cy=246.50072997
        self.x_offset=0.008;  self.y_offset=0.000; self.z_offset=-0.1476
        self.H, self.W = 480, 640

        # 1) 프레임 인덱스 자동 수집
        self.frame_ids = self._scan_frame_ids()
        if self._frame_range is not None:
            s, e = self._frame_range
            self.frame_ids = [f for f in self.frame_ids if s <= f <= e]
        if not self.frame_ids:
            raise RuntimeError(f"No timestamp files found in {self.timestamp_path}")
        self.start_id = self.frame_ids[0]
        self.end_id   = self.frame_ids[-1]

        # 2) 오돔 전체 로드 & 정렬
        self.odom_df = _load_odom_csv(self.odom_csv)
        self.t_odom  = self.odom_df['stamp_sec'].to_numpy(dtype=np.float64)

        # 3) 선속도 추정 & 누적거리 적분
        self.v_lin   = _infer_linear_speed(self.odom_df)            # [m/s]
        self.S_odom  = _cum_arc_length(self.t_odom, self.v_lin)     # [m], S(t)

        # 4) 모든 프레임 TS 배열 (NFS에서 개별 파일 로드가 느려서 로컬 캐시 사용)
        self.t_frames = self._load_timestamps_cached()

        # 5) 카메라 시각에서의 누적 거리 S를 선형보간
        self.S_at_frame = np.interp(self.t_frames, self.t_odom, self.S_odom)  # [m]

        # 6) 최근접 오돔 인덱스(기존 호환용)
        self._build_frame_to_odom_index()

        # 7) SLAM poses 자동 로드 (있으면 get_pose_matrix에서 사용)
        self._slam_poses = None
        self._slam_frame_to_idx = {}
        self._load_slam_poses()

        params = {
            'max_diff': 400,
            'min_diff': 400,
            'iteration': 7,
            'gamma': 1.5,
            'clahe': False,
            'video': False
        }
        self.fieldscale = Fieldscale(**params)

    def __len__(self) -> int:
        return len(self.frame_ids)

    # ------------------ 내부 유틸 ------------------
    def _load_timestamps_cached(self) -> np.ndarray:
        """Load all frame timestamps with local file cache to avoid slow NFS reads."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = self._friendly_override or self.name
        cache_file = os.path.join(cache_dir, f'timestamps_{cache_key}.npy')

        if os.path.exists(cache_file):
            try:
                cached = np.load(cache_file, allow_pickle=True).item()
                if (cached.get('frame_ids') == self.frame_ids
                        and len(cached['t_frames']) == len(self.frame_ids)):
                    return cached['t_frames']
            except (EOFError, ValueError, AttributeError):
                os.remove(cache_file)

        # Load from NFS (slow: ~100s for 1000 files)
        print(f"  Loading {len(self.frame_ids)} timestamps from NFS (first time, caching locally)...",
              flush=True)
        t_frames = np.empty(len(self.frame_ids), dtype=np.float64)
        for i, fid in enumerate(self.frame_ids):
            t_frames[i] = _load_frame_ts(
                os.path.join(self.timestamp_path, f"{fid:05d}.npy"))
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(self.frame_ids)}", flush=True)

        tmp_file = cache_file + '.tmp'
        np.save(tmp_file, {'frame_ids': self.frame_ids, 't_frames': t_frames})
        # np.save appends .npy if not present
        tmp_actual = tmp_file if os.path.exists(tmp_file) else tmp_file + '.npy'
        os.replace(tmp_actual, cache_file)
        print(f"  Cached to {cache_file}", flush=True)
        return t_frames

    def _load_slam_poses(self):
        """Try to load DROID-SLAM poses from gt_poses directory."""
        friendly = self._friendly_override or _WHEEL_FRIENDLY_NAMES.get(self.name)
        if not friendly:
            return

        gt_poses_dir = os.path.join(os.path.dirname(__file__), '..', 'gt_poses')
        slam_dir = os.path.join(gt_poses_dir, f'wheel_{friendly}')
        poses_file = os.path.join(slam_dir, 'poses.npy')
        fids_file = os.path.join(slam_dir, 'frame_ids.npy')

        if os.path.exists(poses_file) and os.path.exists(fids_file):
            self._slam_poses = np.load(poses_file)
            slam_fids = np.load(fids_file)
            self._slam_frame_to_idx = {int(fid): i for i, fid in enumerate(slam_fids)}
            print(f"  Loaded SLAM poses: {len(self._slam_poses)} frames from {slam_dir}")

    def _scan_frame_ids(self):
        if not os.path.isdir(self.timestamp_path):
            return []
        ids = []
        pat = re.compile(r"^(\d{5})\.npy$")
        for name in os.listdir(self.timestamp_path):
            m = pat.match(name)
            if m:
                ids.append(int(m.group(1)))
        ids.sort()
        return ids

    def _build_frame_to_odom_index(self):
        # frame_id → 최근접 odom row index (기존 호환성 유지를 위한 매핑)
        ts = self.t_odom
        idx_right = np.searchsorted(ts, self.t_frames)
        idx_left  = np.clip(idx_right - 1, 0, ts.size - 1)
        idx_right = np.clip(idx_right,      0, ts.size - 1)

        diff_left  = np.abs(ts[idx_left]  - self.t_frames)
        diff_right = np.abs(ts[idx_right] - self.t_frames)
        choose_r   = diff_right < diff_left
        best_idx   = np.where(choose_r, idx_right, idx_left)
        self._odom_index_by_frame = {fid: int(bi) for fid, bi in zip(self.frame_ids, best_idx)}

    # ------------------ 공개 API ------------------
    def load_frame_ts(self, idx: int) -> float:
        return _load_frame_ts(os.path.join(self.timestamp_path, f"{idx:05d}.npy"))

    def load_all_odom(self) -> pd.DataFrame:
        """정렬/보강된 전체 오돔 DataFrame(stamp_sec 포함) 반환."""
        return self.odom_df

    def get_intrinsics(self):
        if self.data_type == "rgb":
            fx=388.90311623
            fy=388.66620522
            cx=318.32120284
            cy=251.50072997
            return fx, fy, cx, cy
        elif self.data_type == "thr":
            fx=404.90311623
            fy=404.66620522
            cx=313.32120284
            cy=249.50072997
            return fx, fy, cx, cy

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera).

        If SLAM poses are available (from DROID-SLAM), uses those.
        Otherwise falls back to straight-line assumption.

        Returns:
            4x4 transformation matrix T_world_camera
        """
        if idx < self.start_id or idx > self.end_id:
            raise KeyError(f"Frame {idx:05d} out of range ({self.start_id:05d}~{self.end_id:05d})")

        # Use SLAM poses if available
        if self._slam_poses is not None and idx in self._slam_frame_to_idx:
            return self._slam_poses[self._slam_frame_to_idx[idx]].copy()

        # Fallback: straight-line motion
        k = self.frame_ids.index(idx)
        S_now = float(self.S_at_frame[k])
        T = np.eye(4, dtype=np.float64)
        T[2, 3] = S_now  # z = forward (camera convention)
        return T

    def get_thr_depth(self, idx):
        pcd = o3d.io.read_point_cloud(f"{self.lidar_path}/{idx:05d}.pcd")    
        points = np.asarray(pcd.points)
        H, W = 512, 640
        depth = np.zeros((H, W), dtype=np.float32)
        fx=404.90311623
        fy=404.66620522
        cx=313.32120284
        cy=249.50072997
        x_offset=0.008
        y_offset=0.000
        z_offset=-0.1078
        
        for point in points:
            x, y, z = point
            x += x_offset
            y += y_offset
            z += z_offset

            if x <= 0:
                continue

            u = int((fx * (-y) / x) + cx)
            v = int((fy * (-z) / x) + cy)

            if 0 <= u < W and 0 <= v < H:
                depth[v, u] = x

        return depth

    def get_depth(self, idx: int):
        return np.load(os.path.join(self.depth_path, f"{idx:05d}.npy"))

    def get_image(self, idx: int):
        return np.load(os.path.join(self.rgb_path, f"{idx:05d}.npy"))

    def get_thr_image(self, idx: int):
        thr_raw = np.load(os.path.join(self.thr_path, f"{idx:05d}.npy")).astype(np.uint16)
        fs = self.fieldscale(thr_raw)
        cmap = plt.get_cmap('binary')
        img = (cmap(fs)[:, :, :3]* 255).astype(np.uint8)
        return img
    # === 신규: 카메라 시각에서의 누적 호 길이/증분 ===
    def get_S_at(self, idx: int) -> float:
        """프레임 idx 시각의 누적 호 길이 S(t_idx) [m]."""
        if idx < self.start_id or idx > self.end_id:
            raise KeyError(f"Frame {idx:05d} out of range ({self.start_id:05d}~{self.end_id:05d})")
        k = self.frame_ids.index(idx)
        return float(self.S_at_frame[k])

    def get_delta_S(self, prev_idx: int, curr_idx: int) -> float:
        """두 프레임 사이 호 길이 ΔS [m] (보간 기반)."""
        return float(self.get_S_at(curr_idx) - self.get_S_at(prev_idx))

    def get_baseline(self, idx: int, clamp_y: bool = True, max_y_ratio: float = 0.3) -> float:
        """Get translation magnitude (baseline) from wheel odometry.

        Args:
            idx: Current frame index (must be >= 1 in frame_ids)
            clamp_y: Unused (for compatibility with BaseDataset)
            max_y_ratio: Unused (for compatibility with BaseDataset)

        Returns:
            Baseline in meters from wheel odometry arc length
        """
        k = self.frame_ids.index(idx)
        if k < 1:
            return 0.0

        prev_idx = self.frame_ids[k - 1]
        return self.get_delta_S(prev_idx, idx)

    def get_gps_baseline(self, idx: int) -> float:
        """Get translation magnitude (baseline) from GPS.

        Args:
            idx: Current frame index (must be >= 1 in frame_ids)

        Returns:
            Baseline in meters from GPS Euclidean distance
        """
        k = self.frame_ids.index(idx)
        if k < 1:
            return 0.0

        prev_idx = self.frame_ids[k - 1]
        return self.get_gps_baseline_between(prev_idx, idx)

    def get_gps_baseline_between(self, prev_idx: int, curr_idx: int) -> float:
        """Get GPS baseline between two frame indices.

        Args:
            prev_idx: Previous frame index
            curr_idx: Current frame index

        Returns:
            Baseline in meters from GPS Euclidean distance
        """
        try:
            gps_curr = self.get_gps(curr_idx)
            gps_prev = self.get_gps(prev_idx)
            # GPS is in NED: [North, East, Down], compute Euclidean distance
            delta = gps_curr - gps_prev
            return float(np.linalg.norm(delta))
        except Exception:
            # Fallback to wheel odometry if GPS not available
            return self.get_delta_S(prev_idx, curr_idx)

    def get_image_size(self) -> tuple:
        """Get image size (H, W)."""
        return (self.H, self.W)

    def get_odom(self, idx: int):
        if idx < self.start_id or idx > self.end_id:
            raise KeyError(f"Frame {idx:05d} out of range ({self.start_id:05d}~{self.end_id:05d})")

        k = self.frame_ids.index(idx)
        S_now   = float(self.S_at_frame[k])

        return {
            "x": S_now,
            "y": 0.0,
        }

    def get_odom_pose(self, idx: int):
        """Get odom position and orientation at frame idx.

        Returns:
            pos_x, pos_y: Position from odometry (in odom frame)
            ori_z, ori_w: Quaternion (only z,w for 2D rotation)
        """
        if idx < self.start_id or idx > self.end_id:
            raise KeyError(f"Frame {idx:05d} out of range ({self.start_id:05d}~{self.end_id:05d})")

        odom_idx = self._odom_index_by_frame[idx]
        row = self.odom_df.iloc[odom_idx]

        return {
            'pos_x': float(row['pos_x']),
            'pos_y': float(row['pos_y']),
            'ori_z': float(row['ori_z']),
            'ori_w': float(row['ori_w']),
        }

    def get_gps(self, idx: int):
        """Get GPS position in NED coordinates.

        If NED folder exists, load from there. Otherwise compute from raw GPS lat/lon/alt.
        """
        ned_path = os.path.join(self.dataset_path, "NED", f"{idx:05d}.npy")
        if os.path.exists(ned_path):
            return np.load(ned_path)

        # Compute NED from raw GPS
        return self._get_gps_as_ned(idx)

    def _get_gps_as_ned(self, idx: int):
        """Convert GPS lat/lon/alt to local NED coordinates."""
        gps_path = os.path.join(self.dataset_path, "GPS", f"{idx:05d}.npy")
        gps_data = np.load(gps_path, allow_pickle=True).item()
        lat, lon, alt = gps_data['lat_lon_alt']

        # Get reference point (first frame)
        if not hasattr(self, '_gps_ref'):
            ref_path = os.path.join(self.dataset_path, "GPS", f"{self.frame_ids[0]:05d}.npy")
            ref_data = np.load(ref_path, allow_pickle=True).item()
            self._gps_ref = ref_data['lat_lon_alt']

        lat_ref, lon_ref, alt_ref = self._gps_ref

        # Earth radius (meters)
        R_earth = 6378137.0

        # Convert to radians
        lat_rad = np.radians(lat)
        lat_ref_rad = np.radians(lat_ref)

        # Local ENU from geodetic (simple flat-earth approximation)
        # Works well for small distances (< few km)
        d_lat = lat - lat_ref
        d_lon = lon - lon_ref
        d_alt = alt - alt_ref

        # North: positive latitude difference
        N = np.radians(d_lat) * R_earth
        # East: positive longitude difference, scaled by cos(lat)
        E = np.radians(d_lon) * R_earth * np.cos(lat_ref_rad)
        # Down: negative altitude difference
        D = -d_alt

        return np.array([N, E, D], dtype=np.float64)

    def has_ned_data(self) -> bool:
        """Check if NED folder exists for this dataset."""
        ned_path = os.path.join(self.dataset_path, "NED")
        return os.path.isdir(ned_path)

    def get(self, idx: int):
        try:
            depth = self.get_depth(idx) if self.data_type=="rgb" else self.get_thr_depth(idx)
            image = self.get_image(idx) if self.data_type=="rgb" else self.get_thr_image(idx)
            odom  = self.get_odom(idx)

            # Load GPS if NED folder exists
            if self.has_ned_data():
                try:
                    gps = self.get_gps(idx)
                except Exception:
                    gps = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            else:
                gps = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            return {
                "image": image,
                "image_og": image,
                "depth": depth,
                "depth_og": depth,
                "position": np.array([odom["x"], odom["y"], 0.0], dtype=np.float64),
                "gps": gps,
                "rpy": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            }
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return None


def _quat_to_yaw(q):
    """Convert quaternion [x, y, z, w] to yaw angle in radians."""
    x, y, z, w = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


class WheelEvalWrapper(BaseDataset):
    """Wrapper for WheelLoader that provides 0-based indexing for evaluation.

    The Evaluator expects datasets to be accessed with 0-based indices (0, 1, 2, ...),
    but WheelLoader uses frame IDs (e.g., 1, 2, 3, ...). This wrapper translates
    between the two.

    Example:
        loader = WheelLoader(name="25_10_20_14_50")
        dataset = WheelEvalWrapper(loader)

        # Now can use 0-based indices
        data = dataset.get(0)  # Gets first frame
        baseline = dataset.get_baseline(1)  # Gets baseline between frame 0 and 1
    """

    def __init__(self, loader: WheelLoader, imu_yaw_offset: float = 54.0,
                 use_imu_rotation: bool = True, use_gps_forward_only: bool = False):
        """Initialize wrapper.

        Args:
            loader: WheelLoader instance
            imu_yaw_offset: Yaw offset (degrees) to apply to IMU orientation for camera-IMU extrinsic.
                           Default 54° calibrated for outdoor dataset with LiDAR GT warp accuracy.
            use_imu_rotation: Whether to use IMU for rotation estimation.
            use_gps_forward_only: If True, use GPS baseline as pure forward motion.
                                If False, use full GPS direction with altitude. Default False.
        """
        self._loader = loader
        self._frame_ids = loader.frame_ids  # List of frame IDs in order
        self._imu_path = os.path.join(loader.dataset_path, "IMU")
        self._imu_yaw_offset = imu_yaw_offset
        self._use_imu_rotation = use_imu_rotation
        self._use_gps_forward_only = use_gps_forward_only

        # Precompute yaw offset rotation matrix
        a = np.radians(imu_yaw_offset)
        c, s = np.cos(a), np.sin(a)
        self._R_yaw_offset = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    @property
    def name(self) -> str:
        """Dataset name for display."""
        return f"Wheel_{self._loader.name}"

    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self._frame_ids)

    def _idx_to_frame_id(self, idx: int) -> int:
        """Convert 0-based index to frame ID."""
        if idx < 0 or idx >= len(self._frame_ids):
            raise IndexError(f"Index {idx} out of range [0, {len(self._frame_ids)})")
        return self._frame_ids[idx]

    def get_intrinsics(self):
        """Get camera intrinsics (fx, fy, cx, cy)."""
        return self._loader.get_intrinsics()

    def get_image_size(self) -> tuple:
        """Get image size (H, W)."""
        return self._loader.get_image_size()

    def get(self, idx: int):
        """Get data for a single frame using 0-based index."""
        frame_id = self._idx_to_frame_id(idx)
        return self._loader.get(frame_id)

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """Get 4x4 camera pose matrix using 0-based index."""
        frame_id = self._idx_to_frame_id(idx)
        return self._loader.get_pose_matrix(frame_id)

    def get_baseline(self, idx: int, clamp_y: bool = True, max_y_ratio: float = 0.3) -> float:
        """Get baseline from wheel odometry using 0-based index.

        Args:
            idx: Current frame index (0-based, must be >= 1)

        Returns:
            Baseline in meters
        """
        if idx < 1:
            return 0.0

        curr_frame_id = self._idx_to_frame_id(idx)
        prev_frame_id = self._idx_to_frame_id(idx - 1)
        return self._loader.get_delta_S(prev_frame_id, curr_frame_id)

    def get_gps_baseline(self, idx: int) -> float:
        """Get baseline from GPS using 0-based index.

        Args:
            idx: Current frame index (0-based, must be >= 1)

        Returns:
            Baseline in meters from GPS Euclidean distance
        """
        if idx < 1:
            return 0.0

        curr_frame_id = self._idx_to_frame_id(idx)
        prev_frame_id = self._idx_to_frame_id(idx - 1)
        return self._loader.get_gps_baseline_between(prev_frame_id, curr_frame_id)

    def _estimate_initial_heading(self) -> float:
        """Estimate initial camera heading from GPS movement direction.

        Uses the first frames with significant movement to estimate
        which direction the camera is facing in NED coordinates.

        Returns:
            Heading in radians (NED convention: 0=North, pi/2=East)
        """
        if hasattr(self, '_initial_heading_cached'):
            return self._initial_heading_cached

        ned_path = os.path.join(self._loader.dataset_path, "NED")

        deltas = []
        for i in range(1, min(50, len(self._frame_ids))):
            curr_id = self._frame_ids[i]
            prev_id = self._frame_ids[i - 1]

            gps_curr = np.load(os.path.join(ned_path, f"{curr_id:05d}.npy"))
            gps_prev = np.load(os.path.join(ned_path, f"{prev_id:05d}.npy"))
            delta = gps_curr - gps_prev

            # Only use frames with significant movement
            if np.linalg.norm(delta[:2]) > 0.01:
                deltas.append(delta[:2])

        if len(deltas) < 3:
            self._initial_heading_cached = 0.0
        else:
            mean_delta = np.mean(deltas, axis=0)
            self._initial_heading_cached = np.arctan2(mean_delta[1], mean_delta[0])

        return self._initial_heading_cached

    def get_relative_pose(self, idx: int):
        """Get relative pose between frame idx-1 and idx.

        Uses GPS for translation (with fallback to wheel odometry).
        Rotation is optionally from IMU (controlled by use_imu_rotation flag).
        IMU orientation is corrected by imu_yaw_offset for camera-IMU extrinsic.

        Returns:
            Tuple of (R, t):
                - R: 3x3 rotation matrix (identity or from IMU)
                - t: 3D translation vector in camera coordinates
        """
        if idx < 1:
            return np.eye(3), np.zeros(3)

        curr_frame_id = self._idx_to_frame_id(idx)
        prev_frame_id = self._idx_to_frame_id(idx - 1)

        # Simple mode: GPS baseline as pure forward motion
        if self._use_gps_forward_only:
            baseline = self.get_gps_baseline(idx)
            t = np.array([0.0, 0.0, baseline], dtype=np.float64)
        else:
            # Full GPS direction estimation with IMU orientation
            t = None
            try:
                gps_curr = self._loader.get_gps(curr_frame_id)
                gps_prev = self._loader.get_gps(prev_frame_id)

                # GPS delta in NED world frame
                delta_ned = gps_curr - gps_prev

                # If GPS delta is essentially zero (no GPS signal, e.g. indoor),
                # fall through to odometry/baseline fallback
                if np.linalg.norm(delta_ned) > 1e-6:
                    # Get IMU rotation at prev frame (body to NED) with yaw offset correction
                    imu_prev = np.load(os.path.join(self._imu_path, f"{prev_frame_id:05d}.npy"), allow_pickle=True).item()
                    q_prev = imu_prev['orientation']
                    R_prev_ned = self._quat_to_rotation_matrix(q_prev)

                    # Apply yaw offset for camera-IMU extrinsic calibration
                    R_prev_corrected = self._R_yaw_offset @ R_prev_ned

                    # Transform delta from NED world to corrected body frame
                    delta_body_ned = R_prev_corrected.T @ delta_ned

                    # Transform from NED body to camera frame (X=right, Y=down, Z=forward)
                    t = S_cv_from_ned @ delta_body_ned

            except Exception as e:
                pass

            # Fallback to wheel odometry if GPS didn't produce a valid translation
            if t is None:
                try:
                    odom_curr = self._loader.get_odom_pose(curr_frame_id)
                    odom_prev = self._loader.get_odom_pose(prev_frame_id)

                    dx = odom_curr['pos_x'] - odom_prev['pos_x']
                    dy = odom_curr['pos_y'] - odom_prev['pos_y']

                    ori_z = odom_prev['ori_z']
                    ori_w = odom_prev['ori_w']
                    yaw_prev = np.arctan2(2 * ori_w * ori_z, 1 - 2 * ori_z * ori_z)

                    cos_yaw = np.cos(-yaw_prev)
                    sin_yaw = np.sin(-yaw_prev)
                    body_x = cos_yaw * dx - sin_yaw * dy
                    body_y = sin_yaw * dx + cos_yaw * dy

                    t = np.array([-body_y, 0.0, body_x], dtype=np.float64)
                except Exception:
                    pass

            # Final fallback: use wheel odometry baseline as pure forward motion
            if t is None:
                baseline = self.get_baseline(idx)
                t = np.array([0.0, 0.0, baseline], dtype=np.float64)

        # Get rotation - identity or from IMU depending on flag
        if self._use_imu_rotation:
            R = self.get_imu_rotation(idx)
        else:
            R = np.eye(3, dtype=np.float64)
        return R, t

    def get_imu_rotation(self, idx: int) -> np.ndarray:
        """Get relative rotation from IMU between frame idx-1 and idx.

        Loads IMU quaternion data and computes relative rotation matrix.
        IMU data is stored as dict with 'orientation' key containing [qx, qy, qz, qw].

        The IMU quaternion is in NED (North-East-Down) frame, so we transform
        the rotation to camera frame (X=Right, Y=Down, Z=Forward).
        Also applies imu_yaw_offset for camera-IMU extrinsic calibration.

        Args:
            idx: Current frame index (0-based, must be >= 1)

        Returns:
            R: 3x3 relative rotation matrix in camera frame (R_curr_prev)
        """
        if idx < 1:
            return np.eye(3, dtype=np.float64)

        try:
            curr_frame_id = self._idx_to_frame_id(idx)
            prev_frame_id = self._idx_to_frame_id(idx - 1)

            # Load IMU data (dict with 'orientation': [qx, qy, qz, qw])
            imu_curr = np.load(os.path.join(self._imu_path, f"{curr_frame_id:05d}.npy"), allow_pickle=True).item()
            imu_prev = np.load(os.path.join(self._imu_path, f"{prev_frame_id:05d}.npy"), allow_pickle=True).item()

            # Extract quaternion from dict
            q_curr = imu_curr['orientation']
            q_prev = imu_prev['orientation']

            # Convert quaternions to rotation matrices (in NED frame)
            R_curr_ned = self._quat_to_rotation_matrix(q_curr)
            R_prev_ned = self._quat_to_rotation_matrix(q_prev)

            # Apply yaw offset for camera-IMU extrinsic calibration
            R_curr_corrected = self._R_yaw_offset @ R_curr_ned
            R_prev_corrected = self._R_yaw_offset @ R_prev_ned

            # Relative rotation in corrected frame
            R_rel_corrected = R_curr_corrected @ R_prev_corrected.T

            # Transform from NED to camera frame
            # Camera: X=Right, Y=Down, Z=Forward
            # NED:    X=North, Y=East,  Z=Down
            # S_cv_from_ned transforms points from NED to camera
            # R_cam = S @ R_ned @ S.T transforms rotation matrix
            R_rel_cam = S_cv_from_ned @ R_rel_corrected @ S_ned_from_cv

            return R_rel_cam

        except Exception as e:
            # If IMU data not available, return identity
            return np.eye(3, dtype=np.float64)

    def _quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.

        Args:
            q: Quaternion [qx, qy, qz, qw]

        Returns:
            R: 3x3 rotation matrix
        """
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]

        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-10:
            return np.eye(3, dtype=np.float64)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float64)

        return R

    def has_imu_data(self) -> bool:
        """Check if IMU data is available for this dataset.

        Returns:
            True if IMU folder exists and has data files
        """
        if not os.path.exists(self._imu_path):
            return False

        # Check if at least one frame has IMU data
        if len(self._frame_ids) > 0:
            first_frame_id = self._frame_ids[0]
            imu_file = os.path.join(self._imu_path, f"{first_frame_id:05d}.npy")
            return os.path.exists(imu_file)

        return False
