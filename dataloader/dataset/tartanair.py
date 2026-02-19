import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from .base import BaseDataset

S_cv_from_ned = np.array([[0,1,0],
                          [0,0,1],
                          [1,0,0]], dtype=float)
S_ned_from_cv = S_cv_from_ned.T  # inverse

class TartanairLoader(BaseDataset):
    def __init__(self, dataset_path, scene, level, num):
        self.dataset_path = dataset_path
        self.scene        = scene
        self.level        = level
        self.num          = num
        
        p_str = f"P{num:03d}"
        
        self.data_path    = os.path.join(self.dataset_path, scene, level, p_str)
        self.poses        = np.loadtxt(f"{self.data_path}/pose_left.txt")

    def __len__(self) -> int:
        return len(self.poses)

    def set_data_path(self):
        self.odom_path = os.path.join(
            self.dataset_path,
            "odom", self.timestamp, self.data_type
        )
        self.depth_path = os.path.join(
            self.dataset_path,
            "proj_depths", self.timestamp, self.data_type, "depth"
        )
        self.image_path = os.path.join(
            self.dataset_path,
            "sync_data", self.timestamp, self.data_type, "img_left"
        )
    
    def get_intrinsics(self):
        return 320, 320, 320, 240

    def get_image_size(self):
        """Get image size (H, W). TartanAir images are 480x640."""
        return (480, 640)
        
    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera).

        Returns:
            4x4 transformation matrix T_world_camera
        """
        C_w = self.poses[idx][:3]  # position in world (NED)
        q_xyzw = self.poses[idx][3:7]  # quaternion (qx, qy, qz, qw)
        q_xyzw = q_xyzw / (np.linalg.norm(q_xyzw) + 1e-12)

        # world <- cam(ned-body)
        R_w_c_ned = R.from_quat(q_xyzw).as_matrix()
        # world <- cam(OpenCV)
        R_w_c_cv = R_w_c_ned @ S_ned_from_cv

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_w_c_cv
        T[:3, 3] = C_w
        return T

    def get_odom(self, idx):
        C_w = self.poses[idx][:3]                      # (tx,ty,tz) world(NED)
        q_xyzw = self.poses[idx][3:7]                  # (qx,qy,qz,qw) 이미 xyzw 순서
        q_xyzw = q_xyzw / (np.linalg.norm(q_xyzw) + 1e-12)

        # world <- cam(ned-body)
        R_w_c_ned = R.from_quat(q_xyzw).as_matrix()

        # world <- cam(OpenCV)  (카메라측 기저변환 적용)
        R_w_c_cv = R_w_c_ned @ S_ned_from_cv
        rpy = R.from_matrix(R_w_c_cv).as_euler('xyz', degrees=False)  # roll,pitch,yaw
        position = C_w
        return position, rpy
    
    def get_depth(self, idx):
        depth = np.load(f"{self.data_path}/depth_left/{idx:06d}_left_depth.npy")
        return depth
    
    def get_image(self, idx):
        img = cv2.imread(f"{self.data_path}/image_left/{idx:06d}_left.png")
        return img
    
    def get_flow(self, idx):
        flow = np.load(f"{self.data_path}/flow/{idx-1:06d}_{idx:06d}_flow.npy")
        return flow
    
    def get(self, idx):
        try:
            position, rpy = self.get_odom(idx)
            depth = self.get_depth(idx)
            image = self.get_image(idx)
            flow  = self.get_flow(idx)
            return {
                "image": image,
                "image_og": image,
                "depth": depth,
                "depth_og": depth,
                "position": position,
                "rpy": rpy,
                "flow": flow
            }
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return None
