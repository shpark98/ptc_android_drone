# -*- coding: utf-8 -*-
import os
import glob
import cv2
import yaml
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from .base import BaseDataset


def _read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _list_cam_images(cam_dir: str):
    data_dir = os.path.join(cam_dir, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    ts = [int(os.path.basename(p).split(".")[0]) for p in files]
    return ts, files

def _load_gt_states(csv_path: str):
    ts, p, q = [], [], []
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            t = int(parts[0])
            px, py, pz = map(float, parts[1:4])
            qw, qx, qy, qz = map(float, parts[4:8])
            ts.append(t)
            p.append([px, py, pz])
            q.append([qw, qx, qy, qz])  # wxyz
    return np.array(ts, dtype=np.int64), np.array(p, dtype=np.float64), np.array(q, dtype=np.float64)

def _nearest_idx(sorted_keys: np.ndarray, key: int) -> int:
    i = np.searchsorted(sorted_keys, key)
    if i <= 0: return 0
    if i >= len(sorted_keys): return len(sorted_keys) - 1
    before = sorted_keys[i-1]; after = sorted_keys[i]
    return i-1 if abs(key - before) <= abs(after - key) else i

def _pose_to_Cw_rpy(T_wc: np.ndarray):
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    # 카메라 중심 C_w = t_wc (이미 world<-cam이면 t_wc가 카메라 원점의 월드좌표)
    C_w = t_wc.copy()
    rpy = R.from_matrix(R_wc).as_euler('xyz', degrees=False)
    return C_w.astype(np.float64), rpy.astype(np.float64)

def _as_SE3(Rm: np.ndarray, t: np.ndarray):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t.reshape(3,)
    return T
def _to_matrix4x4(obj) -> np.ndarray:

    if obj is None:
        return np.eye(4, dtype=np.float64)

    # dict 포맷: {"rows":4, "cols":4, "data":[16개]}
    if isinstance(obj, dict):
        if "data" in obj and "rows" in obj and "cols" in obj:
            r = int(obj["rows"]); c = int(obj["cols"])
            arr = np.asarray(obj["data"], dtype=np.float64).reshape(r, c)
            return arr
        # 혹시 한 단계 감싸진 경우: {"T": {...}}
        for k in ("T", "value", "matrix"):
            if k in obj:
                return _to_matrix4x4(obj[k])

    # 리스트 포맷
    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)

    # 실패 시 항등행렬
    return np.eye(4, dtype=np.float64)
# -----------------------------
# EuRoC Loader
# -----------------------------
class EuRoCLoader(BaseDataset):
    def __init__(self, dataset_path: str, scene: str, cam: str = "cam0",):
        self.scene_root = os.path.join(dataset_path, scene, "mav0")
        self.cam_dir    = os.path.join(self.scene_root, cam)
        self.depth_dir  = os.path.join(self.scene_root, "depth")
        self.cam_yaml   = os.path.join(self.cam_dir, "sensor.yaml")
        self.gt_csv     = os.path.join(self.scene_root, "state_groundtruth_estimate0", "data.csv")

        if not os.path.isfile(self.cam_yaml):
            raise FileNotFoundError(f"sensor.yaml not found: {self.cam_yaml}")

        self.cam_cfg = _read_yaml(self.cam_yaml)
        self.fx, self.fy, self.cx, self.cy = self._parse_intrinsics(self.cam_cfg)
        self.T_cam_from_imu = self._parse_T_cam_from_imu(self.cam_cfg)  # Kalibr: T_BS (Body->Sensor)

        # 이미지 인덱스
        self.img_ts, self.img_paths = _list_cam_images(self.cam_dir)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images under {self.cam_dir}/data")


        self.depth_ts, self.depth_paths = self._index_depth(self.depth_dir)

        self.use_gt = os.path.isfile(self.gt_csv)
        if self.use_gt:
            self.gt_ts, self.gt_p, self.gt_qwxyz = _load_gt_states(self.gt_csv)
        else:
            self.gt_ts = self.gt_p = self.gt_qwxyz = None

    # ---------- Public API ----------
    def __len__(self):
        return len(self.img_paths)

    def get_intrinsics(self):
        return float(self.fx), float(self.fy), float(self.cx), float(self.cy)

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera).

        Returns:
            4x4 transformation matrix T_world_camera
        """
        if not self.use_gt:
            raise RuntimeError("No ground-truth pose available.")

        ts = self.img_ts[idx]
        i = _nearest_idx(self.gt_ts, ts)

        p_w_i = self.gt_p[i]
        qw, qx, qy, qz = self.gt_qwxyz[i]
        R_w_i = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_w_i = _as_SE3(R_w_i, p_w_i)

        # T_w_c = T_w_i @ T_b_c (world <- cam)
        T_w_c = T_w_i @ self.T_cam_from_imu
        return T_w_c

    def get_image(self, idx: int):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.img_paths[idx])
        return img

    def get_depth(self, idx: int):
        if self.depth_ts is None:
            return None
        ts = self.img_ts[idx]
        if ts in self._depth_map:
            path = self._depth_map[ts]
            return np.load(path).astype(np.float32)

        j = _nearest_idx(self.depth_ts, ts)

        return np.load(self.depth_paths[j]).astype(np.float32)

    def get_odom(self, idx: int):
        if not self.use_gt:
            raise RuntimeError("No ground-truth pose available (use_gt=False or file missing).")

        ts = self.img_ts[idx]
        i = _nearest_idx(self.gt_ts, ts)

        p_w_i = self.gt_p[i]           # world<-imu translation
        qw, qx, qy, qz = self.gt_qwxyz[i]
        R_w_i = R.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy: xyzw
        T_w_i = _as_SE3(R_w_i, p_w_i)

        # Body->Cam (Kalibr T_BS): 이미 cam.yaml 안에 4x4 행렬
        T_b_c = self.T_cam_from_imu.copy()

        # world<-cam: T_w_c = T_w_i @ T_b_c
        T_w_c = T_w_i @ T_b_c
        C_w, rpy = _pose_to_Cw_rpy(T_w_c)
        return C_w, rpy

    def get(self, idx: int):
        name = os.path.basename(self.img_paths[idx])
        img  = self.get_image(idx)
        depth = self.get_depth(idx)
        if self.use_gt:
            position, rpy = self.get_odom(idx)
        else:
            position = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            rpy      = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        return {
            "image": img,
            "image_og": img,
            "depth": depth,
            "depth_og": depth,
            "position": position,
            "rpy": rpy,
            "name": name,
            "timestamp_ns": int(self.img_ts[idx]),
        }

    # ---------- Internal ----------
    @staticmethod
    def _parse_intrinsics(cam_cfg: dict):
        """
        Kalibr cam yaml 예시:
          intrinsics: [fu, fv, cu, cy]
          distortion_coeffs: [...]
          T_BS: [ [r11 r12 ... t1], ... ]
        """
        intr = cam_cfg.get("intrinsics", None)
        if intr is None or len(intr) < 4:
            fu = cam_cfg.get("camera_matrix", {}).get("fu", 0.0)
            fv = cam_cfg.get("camera_matrix", {}).get("fv", 0.0)
            cu = cam_cfg.get("camera_matrix", {}).get("cu", 0.0)
            cy = cam_cfg.get("camera_matrix", {}).get("cy", 0.0)
            return fu, fv, cu, cy
        fu, fv, cu, cy = intr[:4]
        return float(fu), float(fv), float(cu), float(cy)


    @staticmethod
    def _parse_T_cam_from_imu(cam_cfg: dict):
        T_raw = cam_cfg.get("T_BS", None)
        T = _to_matrix4x4(T_raw)
        # 안전성: 마지막 행 보정
        T[3, :] = np.array([0, 0, 0, 1], dtype=np.float64)
        return T


    def _index_depth(self, depth_dir: str):
        if not os.path.isdir(depth_dir):
            return None, None
        paths = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
        if len(paths) == 0:
            return None, None
        ts = np.array([int(os.path.basename(p).split(".")[0]) for p in paths], dtype=np.int64)
        self._depth_map = {int(t): p for t, p in zip(ts, paths)}
        return ts, paths
