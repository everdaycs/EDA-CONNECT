import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class EdgeDataset(Dataset):
    def __init__(self, edges_cache_path, image_dir, transform=None):
        self.data = torch.load(edges_cache_path, weights_only=False) # List of all edge items
        self.image_dir = image_dir
        self.transform = transform
        
        # Determine patch size from config but hardcode for now or pass in
        self.patch_size = (128, 64) # W, H
        
        # Albumentations
        if transform is None:
            self.aug = A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # ToTensorV2 handled manually or outside
            ])

    def __len__(self):
        return len(self.data)
        
    def _crop_corridor(self, img, u_bbox, v_bbox):
        # Merge bboxes
        x1 = min(u_bbox[0], v_bbox[0])
        y1 = min(u_bbox[1], v_bbox[1])
        x2 = max(u_bbox[2], v_bbox[2])
        y2 = max(u_bbox[3], v_bbox[3])
        
        # Padding logic
        w = x2 - x1
        h = y2 - y1
        diag = np.sqrt(w**2 + h**2)
        pad = max(16, int(0.1 * diag))
        
        px1 = max(0, int(x1 - pad))
        py1 = max(0, int(y1 - pad))
        px2 = min(img.shape[1], int(x2 + pad))
        py2 = min(img.shape[0], int(y2 + pad))
        
        patch = img[py1:py2, px1:px2]
        
        # Resize to fixed input size
        if patch.size == 0:
             patch = np.zeros((self.patch_size[1], self.patch_size[0], 3), dtype=np.uint8)
        else:
             patch = cv2.resize(patch, self.patch_size)
             
        # TODO: Mask out component internals?
        # For simple version, just use raw pixels.
        
        return patch

    def _get_geom_feats(self, u_bbox, v_bbox):
        # Normalize coords relative to each other?
        # Absolute coords are not useful for CNN
        # Relative vector:
        ucx, ucy = (u_bbox[0]+u_bbox[2])/2, (u_bbox[1]+u_bbox[3])/2
        vcx, vcy = (v_bbox[0]+v_bbox[2])/2, (v_bbox[1]+v_bbox[3])/2
        
        dx = vcx - ucx
        dy = vcy - ucy
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # Simple geometric vector
        return np.array([dx, dy, dist, angle, 
                         u_bbox[2]-u_bbox[0], u_bbox[3]-u_bbox[1], 
                         v_bbox[2]-v_bbox[0], v_bbox[3]-v_bbox[1]], dtype=np.float32)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item['img_name'])
        
        # Optimization: Use LRU Cache for images if dataset is small enough
        img = cv2.imread(img_path)
        if img is None:
            # Fallback
            img = np.zeros((1000, 1000, 3), dtype=np.uint8)
            
        # 1. Crop Patch
        patch = self._crop_corridor(img, item['u_bbox'], item['v_bbox'])
        
        # 2. Augment
        augmented = self.aug(image=patch)['image']
        # Convert to Tensor (C, H, W)
        patch_tensor = torch.from_numpy(augmented.transpose(2, 0, 1)).float()
        
        # 3. Geom Features
        geom = torch.from_numpy(self._get_geom_feats(item['u_bbox'], item['v_bbox']))
        
        # 4. Label
        label = item['label'] # 0 or 1
        
        return patch_tensor, geom, torch.tensor(label, dtype=torch.float)
