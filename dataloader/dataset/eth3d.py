# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base import BaseDataset


class ETH3DLoader(BaseDataset):
    def __init__(self, dataset_path: str, scene: str):
        self.dataset_path = os.path.join(dataset_path, "undistorted", scene)
        self.depth_path = os.path.join(dataset_path, "gt", scene, "ground_truth_depth", "dslr_images")
        self.calib_root = os.path.join(self.dataset_path, "dslr_calibration_undistorted")
        self.images_txt = os.path.join(self.calib_root, "images.txt")
        self.cameras_txt = os.path.join(self.calib_root, "cameras.txt")
        self.images_dir = os.path.join(self.dataset_path, "images")

        # 파싱
        self._image_names, self._poses, self._cam_ids = self._parse_images_txt(self.images_txt)
        self._intr_by_camid = self._parse_cameras_txt(self.cameras_txt)

        # 기본 intrinsics: 첫 번째 프레임의 camera_id 사용
        if len(self._cam_ids) == 0:
            raise RuntimeError("No camera IDs found in images.txt")
        self._default_camid = self._cam_ids[0]

    def __len__(self) -> int:
        return len(self._image_names)

    # -----------------------------
    # Public API
    # -----------------------------
    def get_intrinsics(self):
        """
        반환: fx, fy, cx, cy (float)
        """
        intr = self._intr_by_camid.get(self._default_camid, None)
        if intr is None:
            return 1000.0, 1000.0, 0.0, 0.0
        return intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera).

        Returns:
            4x4 transformation matrix T_world_camera
        """
        q_xyzw = self._poses[idx]["q_xyzw"]
        t_cw = self._poses[idx]["t_cw"]

        # COLMAP: R_cw is world->cam rotation
        R_cw = R.from_quat([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]]).as_matrix()
        R_wc = R_cw.T  # world <- cam
        C_w = -R_cw.T @ t_cw  # camera center in world

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_wc
        T[:3, 3] = C_w
        return T

    def get_odom(self, idx: int):
        """
        반환:
          - position: (3,) 카메라 중심 C_w (world 좌표계)
          - rpy: (3,) roll, pitch, yaw (radians), world<-cam 회전의 오일러('xyz')
        """
        q_xyzw = self._poses[idx]["q_xyzw"]    # (qx, qy, qz, qw)
        t_cw   = self._poses[idx]["t_cw"]      # (tx, ty, tz), world->cam 변환의 translation
        # COLMAP convention:
        #   x_cam = R_cw * X_world + t_cw
        #   R_cw = quat(qw,qx,qy,qz) as rotmat (world->cam)
        #   Camera center C_w = -R_cw^T * t_cw
        R_cw = R.from_quat([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]]).as_matrix()
        C_w  = -R_cw.T @ t_cw
        R_wc = R_cw.T  # world<-cam
        rpy  = R.from_matrix(R_wc).as_euler('xyz', degrees=False)
        return C_w, rpy

    def get_image(self, idx: int):
        """
        반환: BGR np.ndarray
        """
        name = self._image_names[idx]
        img_path = os.path.join(self.images_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img
    
    def get_depth(self, idx: int):
        """
        반환: 깊이 맵 np.ndarray (float32)
        """
        name = self._image_names[idx]
        depth_name = name.split("/")[-1]
        depth_path = os.path.join(self.depth_path, depth_name)
        if not os.path.isfile(depth_path):
            raise FileNotFoundError(f"Depth map not found: {depth_path}")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth map not found: {depth_path}")
        return depth

    def get(self, idx: int):
        try:
            position, rpy = self.get_odom(idx)
            image = self.get_image(idx)
            # depth = self.get_depth(idx)
            return {
                "image": image,
                "image_og": image,
                # "depth": depth, 
                # "depth_og": depth,
                "position": position,
                "rpy": rpy,
                "name": self._image_names[idx],
            }
        except Exception as e:
            print(f"[ETH3DLoader] Error at idx={idx}: {e}")
            return None

    # -----------------------------
    # Internal parsers
    # -----------------------------
    @staticmethod
    def _parse_images_txt(images_txt: str):
        """
        COLMAP images.txt 파싱.
        반환:
          - image_names: [N] 이미지 상대경로 (보통 'dslr_images_undistorted/xxx.JPG')
          - poses: list of dict: {'q_xyzw': (4,), 't_cw': (3,)}
          - cam_ids: [N] camera_id
        """
        if not os.path.isfile(images_txt):
            raise FileNotFoundError(f"images.txt not found: {images_txt}")

        image_names = []
        poses = []
        cam_ids = []

        with open(images_txt, "r") as f:
            raw = [l.strip() for l in f.readlines()]
        # 주석/빈줄 제거
        lines = [l for l in raw if l and not l.startswith("#")]

        # images.txt는 이미지당 2줄 (헤더 + 2D 포인트들)
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 10:
                continue
            # 포맷: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            # 단, 사용자는 앞서 예시로 'QW QX QY QZ' 순서를 확인해줌.
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = parts[9]

            # 우리 내부 저장은 (qx,qy,qz,qw)로 통일
            q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
            q_xyzw /= (np.linalg.norm(q_xyzw) + 1e-12)
            t_cw = np.array([tx, ty, tz], dtype=np.float64)

            image_names.append(name)
            poses.append({"q_xyzw": q_xyzw, "t_cw": t_cw})
            cam_ids.append(cam_id)

        return image_names, poses, cam_ids

    @staticmethod
    def _parse_cameras_txt(cameras_txt: str):
        if not os.path.isfile(cameras_txt):
            return {}

        intrinsics = {}
        with open(cameras_txt, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
                if len(parts) < 4:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                w = int(parts[2]); h = int(parts[3])
                params = list(map(float, parts[4:]))

                fx = fy = cx = cy = 0.0
                if model == "PINHOLE":
                    # fx fy cx cy
                    if len(params) >= 4:
                        fx, fy, cx, cy = params[:4]
                elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                    # f cx cy [k1 k2]
                    if len(params) >= 3:
                        f, cx, cy = params[:3]
                        fx = fy = f
                else:
                    # 그 외 모델은 우선 f만 있으면 fx=fy=f, cx,cy는 0으로 fallback
                    if len(params) >= 1:
                        f = params[0]
                        fx = fy = f

                intrinsics[cam_id] = {
                    "fx": float(fx), "fy": float(fy),
                    "cx": float(cx), "cy": float(cy),
                    "model": model, "w": w, "h": h,
                }
        return intrinsics
