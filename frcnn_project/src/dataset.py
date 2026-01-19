import torch
from torch.utils.data import Dataset
import cv2
import json
import numpy as np

class SchematicDataset(Dataset):
    def __init__(self, split_file, split='train', transforms=None, input_size=None):
        with open(split_file, 'r') as f:
            data = json.load(f)
        
        self.img_paths = data[split]
        self.annotations = data['annotations']
        self.transforms = transforms
        self.split = split
        self.input_size = input_size # [h, w] tuple or None
        self.crop_size = 800  # Size for random crop

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get cached boxes
        annots = self.annotations[img_path]
        
        boxes = []
        labels = []
        for ann in annots:
            boxes.append(ann['bbox'])
            # Shift class_id by 1 because 0 is background in FasterRCNN
            labels.append(ann['class_id'] + 1)
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Random Crop Logic for Training
        if self.split == 'train':
            h, w = img.shape[:2]
            if h > self.crop_size and w > self.crop_size:
                # Attempt to crop a region containing objects
                for _ in range(50): # Retry 50 times to find a valid crop
                    top = np.random.randint(0, h - self.crop_size)
                    left = np.random.randint(0, w - self.crop_size)
                    
                    bottom = top + self.crop_size
                    right = left + self.crop_size
                    
                    # Filter boxes inside crop
                    # Center of box should be inside crop
                    box_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
                    box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
                    
                    mask = (box_centers_x > left) & (box_centers_x < right) & \
                           (box_centers_y > top) & (box_centers_y < bottom)
                           
                    if mask.sum() > 0: # Found valid objects
                        # Crop Image
                        img = img[top:bottom, left:right]
                        
                        # Adjust Boxes
                        boxes = boxes[mask]
                        labels = labels[mask]
                        
                        boxes[:, 0] -= left
                        boxes[:, 2] -= left
                        boxes[:, 1] -= top
                        boxes[:, 3] -= top
                        
                        # Clip boxes to crop boundaries
                        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.crop_size)
                        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.crop_size)
                        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.crop_size)
                        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.crop_size)
                        
                        break
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([], dtype=torch.float32)
            
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # Custom transform implementation or use albumentations/torchvision
            # For simplicity in this demo, implementation usually goes here.
            # However, since torchvision fasterrcnn handles internal resizing,
            # we mainly care about photometric distortions here.
            # To strictly follow "torchvision priority", we return PIL or Tensor.
            pass
            
        # Convert to tensor [0, 1] C,H,W
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
