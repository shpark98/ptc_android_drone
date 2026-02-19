import os
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .base import BaseDataset

S_cv_from_ned  = np.array([[0,1,0],
                           [0,0,1],
                           [1,0,0]], dtype=float)
S_ned_from_cv  = S_cv_from_ned.T  # inverse


class CampusLoader(BaseDataset):
    def __init__(self):
        self.dataset_path = "/home/nas/Dataset3/data"
        self.lidar_path   = os.path.join(self.dataset_path, "LiDAR")
        self.rgb_path     = os.path.join(self.dataset_path, "top")
        self.syn_path     = os.path.join(self.dataset_path, "raw_temp", "raw_mid")
        self.fx=408.90311623
        self.fy=408.66620522
        self.cx=309.32120284
        self.cy=246.50072997
        
        self.x_offset=0.008
        self.y_offset=0.000
        self.z_offset=-0.1476
        
        self.H, self.W = 480, 640
        self._num_frames = self._count_frames()

    def __len__(self) -> int:
        return self._num_frames

    def _count_frames(self) -> int:
        """Count available frames by checking rgb directory."""
        if not os.path.isdir(self.rgb_path):
            return 0
        return len([f for f in os.listdir(self.rgb_path) if f.endswith('.npy')])

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera).

        Returns:
            4x4 transformation matrix T_world_camera
        """
        path = os.path.join(self.dataset_path, "odoom3", f"{idx:05d}.txt")
        pose_mat = np.loadtxt(path).astype(np.float64).reshape(4, 4)

        # Convert rotation from NED to OpenCV frame
        rotation = pose_mat[:3, :3] @ S_ned_from_cv

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3, 3] = pose_mat[:3, 3]
        return T

    def get_depth(self, idx):
        pcd = o3d.io.read_point_cloud(f"{self.lidar_path}/{idx:05d}.pcd")    
        points = np.asarray(pcd.points)

        depth = np.zeros((self.H, self.W), dtype=np.float32)

        for point in points:
            x, y, z = point
            x += self.x_offset
            y += self.y_offset
            z += self.z_offset

            if x <= 0:
                continue

            u = int((self.fx * (-y) / x) + self.cx)
            v = int((self.fy * (-z) / x) + self.cy)

            if 0 <= u < self.W and 0 <= v < self.H:
                depth[v, u] = x

        return depth
    
    def get_odom(self, idx):
        path = os.path.join(self.dataset_path, "odoom3", f"{idx:05d}.txt")
        pose_mat = np.loadtxt(path).astype(np.float32).reshape(4, 4)
        
        position = pose_mat[:3, 3]
        rotation = pose_mat[:3, :3]
        
        rotation = rotation @ S_ned_from_cv  # world <- cam(OpenCV)  (카메라측 기저변환 적용)
        # position = np.array([position[2], position[0], position[1]])  
        
        rpy = R.from_matrix(rotation).as_euler('xyz', degrees=False)
        return position, rpy
    
    def get_image(self, idx):
        img = np.load(f"{self.rgb_path}/{idx:05d}.npy")
        return img
    
    def get_synthetic_image(self, idx):
        img = np.load(f"{self.syn_path}/{idx:05d}.npy")
        return img
    
    def get(self, idx):
        try:
            position, rpy = self.get_odom(idx)
            depth = self.get_depth(idx)
            image = self.get_image(idx)
            return {
                "image": image,
                "image_og": image,
                "mid": self.get_synthetic_image(idx),
                "depth": depth,
                "depth_og": depth,
                "position": position,
                "rpy": rpy,
            }
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return None