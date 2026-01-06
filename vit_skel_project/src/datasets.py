import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, img_dir, skel_dir, img_size=512, transform=None):
        self.img_dir = img_dir
        self.skel_dir = skel_dir
        self.img_size = img_size
        self.transform = transform
        
        self.images = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        # Filter only ones with skeletons
        self.samples = []
        for p in self.images:
            name = os.path.basename(p)
            stem = os.path.splitext(name)[0]
            # Try png for mask
            sk_p = os.path.join(skel_dir, stem + ".png")
            if os.path.exists(sk_p):
                self.samples.append((p, sk_p))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_p, sk_p = self.samples[idx]
        
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(sk_p, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        # To Tensor
        img = torch.from_numpy(img).permute(2, 0, 1) # C,H,W
        mask = torch.from_numpy(mask).unsqueeze(0)   # 1,H,W
        
        return img, mask
