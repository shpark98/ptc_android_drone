import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from .utils.fieldscale import Fieldscale
from .base import BaseDataset

S_cv_from_ned  = np.array([[0,1,0],
                           [0,0,1],
                           [1,0,0]], dtype=float)
S_ned_from_cv  = S_cv_from_ned.T  # inverse


class MS2Loader(BaseDataset):
    def __init__(self, dataset_path, timestamp, data_type="thr"):
        self.dataset_path = dataset_path
        self.timestamp = timestamp
        self.data_type = data_type
        self.set_data_path()
        params = {
            'max_diff': 400,
            'min_diff': 400,
            'iteration': 7,
            'gamma': 1.5,
            'clahe': False,
            'video': False
        }
        self.fieldscale = Fieldscale(**params)
        self._num_frames = self._count_frames()

    def __len__(self) -> int:
        return self._num_frames

    def _count_frames(self) -> int:
        """Count available frames by checking depth directory."""
        if not os.path.isdir(self.depth_path):
            return 0
        return len([f for f in os.listdir(self.depth_path) if f.endswith('.png')])

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
        path = os.path.join(self.dataset_path, "sync_data", self.timestamp, "calib.npy")
        calibs = np.load(path, allow_pickle=True).item()

        intrinsics = calibs[f"K_{self.data_type}L"]
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        return fx, fy, cx, cy

    def get_image_size(self):
        """Get image size (H, W) by reading the first frame."""
        sample = self.get(0)
        if sample is None:
            return (512, 640)  # Default MS2 size
        return sample['image'].shape[:2]

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera).

        Returns:
            4x4 transformation matrix T_world_camera
        """
        path = os.path.join(self.odom_path, f"{idx:06d}.txt")
        pose_mat = np.loadtxt(path).astype(np.float64).reshape(4, 4)
        return pose_mat

    def get_odom(self, idx):
        path = os.path.join(self.odom_path, f"{idx:06d}.txt")
        pose_mat = np.loadtxt(path).astype(np.float32).reshape(4, 4)

        position = pose_mat[:3, 3]
        rotation = pose_mat[:3, :3]

        rpy = R.from_matrix(rotation).as_euler('xyz', degrees=False)
        return position, rpy
    
    def get_depth(self, idx):
        path = os.path.join(self.depth_path, f"{idx:06d}.png")
        depth = np.array(Image.open(path)).squeeze().astype(np.float32) / 256.0
        return depth
    
    def get_image(self, idx):
        path = os.path.join(self.image_path, f"{idx:06d}.png")
    
        if self.data_type == "thr":
            thr = np.array(Image.open(path)).astype(np.uint16)
            fs = self.fieldscale(thr)
            img = np.stack((fs, fs, fs), axis=-1)

            return img, thr
        
        else:
            img = cv2.imread(path)

        return img
    
    def get(self, idx):
        try:
            position, rpy = self.get_odom(idx)
            depth = self.get_depth(idx)
            if self.data_type == "thr":
                image, thermal = self.get_image(idx)
            else:
                image = self.get_image(idx)
            
            return {
                "image": image,
                "image_og": image,
                "depth": depth,
                "depth_og": depth,
                "position": position,
                "rpy": rpy,
                "thermal": thermal if self.data_type == "thr" else None
            }
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return None