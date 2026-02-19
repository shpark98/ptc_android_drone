# -*- coding: utf-8 -*-
import os, glob, cv2, yaml, re
import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy.spatial.transform import Rotation as R

from .base import BaseDataset


def _read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _name_to_ts_ns(filename_noext: str) -> int | None:
    m = re.search(r"(\d+(?:\.\d+)?)", filename_noext)
    if not m:
        return None
    token = m.group(1)
    if "." in token:  # float seconds
        sec = float(token)
        return int(round(sec * 1e9))
    # integer-like
    x = int(token)
    L = len(token)
    if L >= 15:              # ns
        return x
    elif L >= 13:            # ms
        return x * 1_000_000
    elif L >= 10:            # s
        return x * 1_000_000_000
    else:
        return None

def _list_ts_files(dirpath: str, exts: Tuple[str, ...]) -> Tuple[np.ndarray, List[str]]:
    if not os.path.isdir(dirpath):
        print(f"[warn] directory not found: {dirpath}")
        return np.empty((0,), np.int64), []

    paths: List[str] = []
    for ext in exts:
        ext = ext.lstrip(".")
        paths.extend(glob.glob(os.path.join(dirpath, f"*.{ext}")))
    if not paths:
        print(f"[warn] no files with {exts} in {dirpath}")
        return np.empty((0,), np.int64), []

    ts_ns_list, good_paths = [], []
    for p in sorted(paths):
        base = os.path.splitext(os.path.basename(p))[0]
        t_ns = _name_to_ts_ns(base)
        if t_ns is not None:
            ts_ns_list.append(t_ns)
            good_paths.append(p)

    if not ts_ns_list:
        print(f"[warn] no parsable timestamps in filenames under {dirpath}")
        return np.empty((0,), np.int64), []

    idx = np.argsort(np.asarray(ts_ns_list, dtype=np.int64))
    ts_sorted    = np.asarray([ts_ns_list[i] for i in idx], dtype=np.int64)
    paths_sorted = [good_paths[i] for i in idx]
    return ts_sorted, paths_sorted

def _nearest_idx(sorted_keys: np.ndarray, key: int) -> int:
    if len(sorted_keys) == 0:
        raise IndexError("empty keys")
    i = np.searchsorted(sorted_keys, key)
    if i <= 0: return 0
    if i >= len(sorted_keys): return len(sorted_keys) - 1
    before = sorted_keys[i-1]; after = sorted_keys[i]
    return i-1 if abs(key - before) <= abs(after - key) else i

def _pose_to_Cw_rpy(T_wc: np.ndarray):
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    C_w = t_wc.copy()
    rpy = R.from_matrix(R_wc).as_euler('xyz', degrees=False)
    return C_w.astype(np.float64), rpy.astype(np.float64)

def _as_SE3(Rm: np.ndarray, t: np.ndarray):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3]  = t.reshape(3,)
    return T

def _load_array_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        d = np.load(path)
        for k in ("arr", "data", "depth", "array", "arr_0"):
            if k in d: return d[k]
        keys = list(d.keys())
        return d[keys[0]] if keys else None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def _to_matrix4x4(obj) -> np.ndarray:
    if obj is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(obj, dict):
        if "data" in obj and "rows" in obj and "cols" in obj:
            r = int(obj["rows"]); c = int(obj["cols"])
            return np.asarray(obj["data"], dtype=np.float64).reshape(r, c)
        for k in ("T", "value", "matrix"):
            if k in obj: return _to_matrix4x4(obj[k])
    if isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj, dtype=np.float64)
        if arr.shape == (4, 4): return arr
        if arr.size == 16: return arr.reshape(4, 4)
    return np.eye(4, dtype=np.float64)

def _load_pose_matrix_file(path: str) -> np.ndarray:
    with open(path, "r") as f:
        nums = [float(x) for x in re.split(r"[,\s]+", f.read().strip()) if x]
    if len(nums) == 12:
        M = np.eye(4, dtype=np.float64)
        M[:3, :4] = np.asarray(nums, dtype=np.float64).reshape(3, 4)
        M[3, :] = [0, 0, 0, 1]
        return M
    elif len(nums) == 16:
        M = np.asarray(nums, dtype=np.float64).reshape(4, 4)
        M[3, :] = [0, 0, 0, 1]
        return M
    else:
        raise ValueError(f"pose file '{os.path.basename(path)}' has {len(nums)} values (expected 12 or 16)")

def _load_K_txt(path: str) -> Optional[np.ndarray]:
    """K.txt에서 3x3/3x4/4x4 형태를 읽어 3x3 K 반환."""
    if not os.path.isfile(path): return None
    nums = []
    with open(path, "r") as f:
        for tok in re.split(r"[,\s]+", f.read().strip()):
            if tok: nums.append(float(tok))
    arr = np.asarray(nums, dtype=np.float64)
    if arr.size in (9, 12, 16):
        # 3x3 / 3x4 / 4x4 모두 reshape 뒤 상위 3x3 사용
        if arr.size == 9:  M = arr.reshape(3,3)
        elif arr.size == 12: M = arr.reshape(3,4)[:,:3]
        else: M = arr.reshape(4,4)[:3,:3]
        return M
    # 줄단위로 3개씩 끊어져 있을 수도 있음 -> 3x3 시도
    lines = [ln for ln in open(path).read().strip().splitlines() if ln.strip()]
    if len(lines) >= 3:
        rows = []
        for ln in lines[:3]:
            vals = [float(x) for x in re.split(r"[,\s]+", ln.strip()) if x]
            rows.append(vals)
        M = np.asarray(rows, dtype=np.float64)
        if M.shape[0] >= 3 and M.shape[1] >= 3:
            return M[:3,:3]
    return None

# -----------------------------
# VOID Loader
# -----------------------------
class VOIDLoader(BaseDataset):
    def __init__(self, dataset_path: str, scene: str, num: int = 150, *, pose_is_c_w: bool = False):
        self.scene_root = os.path.join(dataset_path, f"void_{num}", "data", scene)
        if not os.path.isdir(self.scene_root):
            raise FileNotFoundError(self.scene_root)

        self.pose_is_c_w = pose_is_c_w

        self.img_dir   = os.path.join(self.scene_root, "image")
        self.gt_dir    = os.path.join(self.scene_root, "ground_truth")
        self.sd_dir    = os.path.join(self.scene_root, "sparse_depth")
        self.vm_dir    = os.path.join(self.scene_root, "validity_map")
        self.pose_dir  = os.path.join(self.scene_root, "absolute_pose")
        self.k_txt     = os.path.join(self.scene_root, "K.txt")

        # ---- intrinsics 우선순위: K.txt -> intrinsics.yaml -> NaN
        self.fx = self.fy = self.cx = self.cy = np.nan
        K = _load_K_txt(self.k_txt)
        if K is None:
            # 예비: image/K.txt도 시도
            K = _load_K_txt(os.path.join(self.img_dir, "K.txt"))
        if K is not None:
            self.fx, self.fy, self.cx, self.cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
        elif os.path.isfile(self.int_yaml):
            cfg = _read_yaml(self.int_yaml)
            for k in ("fx","fy","cx","cy"):
                if k in cfg: setattr(self, k, float(cfg[k]))
            if np.isnan(self.fx) or np.isnan(self.fy) or np.isnan(self.cx) or np.isnan(self.cy):
                intr = cfg.get("intrinsics", None)
                if intr and len(intr) >= 4:
                    self.fx, self.fy, self.cx, self.cy = map(float, intr[:4])

        # index images
        self.img_ts, self.img_paths = _list_ts_files(self.img_dir, (".png",".jpg",".jpeg"))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images under {self.img_dir}")

        # optional: dense GT / sparse depth / validity
        self.gt_ts, self.gt_paths = _list_ts_files(self.gt_dir, (".png",".npy",".npz")) if os.path.isdir(self.gt_dir) else (np.empty((0,),np.int64), [])
        self.sd_ts, self.sd_paths = _list_ts_files(self.sd_dir, (".png",".npy",".npz")) if os.path.isdir(self.sd_dir) else (np.empty((0,),np.int64), [])
        self.vm_ts, self.vm_paths = _list_ts_files(self.vm_dir, (".png",".npy",".npz")) if os.path.isdir(self.vm_dir) else (np.empty((0,),np.int64), [])

        # poses
        self.has_pose, self.pose_ts, self.pose_data = self._index_poses(self.pose_dir)

    def __len__(self): return len(self.img_paths)

    def get_intrinsics(self) -> Tuple[float,float,float,float]:
        return float(self.fx), float(self.fy), float(self.cx), float(self.cy)

    def get_image(self, idx: int) -> np.ndarray:
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.img_paths[idx])
        return img

    def get_dense_depth(self, idx: int) -> Optional[np.ndarray]:
        if len(self.gt_paths) == 0: return None
        ts = self.img_ts[idx]
        j  = _nearest_idx(self.gt_ts, ts)
        arr = _load_array_any(self.gt_paths[j])
        if arr is None: return None
        return arr.astype(np.float32, copy=False)

    def get_sparse_depth(self, idx: int) -> Optional[np.ndarray]:
        if len(self.sd_paths) == 0: return None
        ts = self.img_ts[idx]
        j  = _nearest_idx(self.sd_ts, ts)
        arr = _load_array_any(self.sd_paths[j])
        return arr.astype(np.float32, copy=False) if isinstance(arr, np.ndarray) else None

    def get_validity(self, idx: int) -> Optional[np.ndarray]:
        if len(self.vm_paths) == 0: return None
        ts = self.img_ts[idx]
        j  = _nearest_idx(self.vm_ts, ts)
        arr = _load_array_any(self.vm_paths[j])
        return (arr > 0).astype(np.uint8) if isinstance(arr, np.ndarray) else None

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera).

        Returns:
            4x4 transformation matrix T_world_camera
        """
        if not self.has_pose:
            raise RuntimeError("No absolute pose available.")
        ts = self.img_ts[idx]
        j = _nearest_idx(self.pose_ts, ts)
        return self._pose_at_index(j)

    def get_odom(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_pose:
            raise RuntimeError("No absolute pose available.")
        ts = self.img_ts[idx]
        j  = _nearest_idx(self.pose_ts, ts)
        T_w_c = self._pose_at_index(j)
        C_w, rpy = _pose_to_Cw_rpy(T_w_c)
        return C_w, rpy

    def get(self, idx: int) -> Dict:
        name = os.path.basename(self.img_paths[idx])
        img  = self.get_image(idx)
        z_gt = self.get_dense_depth(idx)
        z_sd = self.get_sparse_depth(idx)
        vm   = self.get_validity(idx)

        if self.has_pose:
            position, rpy = self.get_odom(idx)
        else:
            position = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            rpy      = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        return {
            "image": img,
            "image_og": img,
            "depth": z_gt/256.0,
            "depth_og": z_gt/256.0,
            "sparse_depth": z_sd,
            "validity_map": vm,
            "position": position,
            "rpy": rpy,
            "name": name,
        }

    # ---------- internal: poses ----------
    def _index_poses(self, pose_dir: str):
        if not os.path.isdir(pose_dir):
            return False, np.empty((0,),np.int64), None

        # (A) 타임스탬프별 파일(3x4/4x4)
        ts_pf, paths_pf = _list_ts_files(pose_dir, (".txt",".dat",".pose"))
        if len(paths_pf) > 0:
            self._pose_mode       = "per_file_mat"
            self._pose_mats_ts    = ts_pf
            self._pose_mats_paths = paths_pf
            self._pose_mats_cache: Dict[int, np.ndarray] = {}
            return True, ts_pf, {"mode": "per_file_mat"}

        # (B) 단일 테이블 파일
        single = []
        for ext in (".txt",".csv",".npy",".npz"):
            single.extend(glob.glob(os.path.join(pose_dir, f"*{ext}")))
        single = sorted(single)
        if len(single) > 0:
            path = single[0]
            ext  = os.path.splitext(path)[1].lower()
            if ext in (".npy",".npz"):
                data = np.load(path, allow_pickle=True)
                arr  = np.asarray(data)
            else:
                rows = []
                with open(path,"r") as f:
                    for line in f:
                        line=line.strip()
                        if not line or line.startswith("#"): continue
                        parts = re.split(r"[,\s]+", line)
                        rows.append([float(x) for x in parts])
                arr = np.asarray(rows, dtype=np.float64)

            ts, table = self._parse_pose_table_from_array(arr)
            self._pose_table = table
            self._pose_mode  = "table"
            return True, ts, table

        return False, np.empty((0,),np.int64), None

    def _parse_pose_table_from_array(self, arr: np.ndarray):
        if arr.ndim != 2:
            raise ValueError(f"pose table ndim must be 2, got {arr.shape}")

        N, _ = arr.shape

        ts = arr[:, 0].astype(np.int64)
        t  = arr[:, 1:4].astype(np.float64)
        q  = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (N,1))
        return ts, {"ts": ts, "t": t, "q": q}
    
    def _pose_at_index(self, j: int) -> np.ndarray:
        mode = getattr(self, "_pose_mode", None)
        if mode == "table":
            t = self._pose_table["t"][j]
            q_xyzw = self._pose_table["q"][j]
            R_wc = R.from_quat(q_xyzw).as_matrix()
            T = _as_SE3(R_wc, t)
            return np.linalg.inv(T) if self.pose_is_c_w else T

        if mode == "per_file_mat":
            ts = self._pose_mats_ts[j]
            path = self._pose_mats_paths[j]
            if not hasattr(self, "_pose_mats_cache"):
                self._pose_mats_cache = {}
            if ts in self._pose_mats_cache:
                T = self._pose_mats_cache[ts]
            else:
                T = _load_pose_matrix_file(path)
                self._pose_mats_cache[ts] = T
            return np.linalg.inv(T) if self.pose_is_c_w else T

        raise RuntimeError("pose mode is undefined but has_pose=True")
