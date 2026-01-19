import os
import json
import glob
import cv2
import numpy as np
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def yolo_to_xyxy(x_center, y_center, w, h, width, height):
    x1 = (x_center - w/2) * width
    y1 = (y_center - h/2) * height
    x2 = (x_center + w/2) * width
    y2 = (y_center + h/2) * height
    return [x1, y1, x2, y2]

def sanity_check(args):
    data_root = Path(args.data_root)
    img_dir = data_root / args.img_dir_name
    lbl_dir = data_root / args.lbl_dir_name
    
    if not img_dir.exists() or not lbl_dir.exists():
        logger.error(f"Directories not found: {img_dir} or {lbl_dir}")
        return

    # Gather files
    img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    logger.info(f"Found {len(img_files)} images.")

    valid_samples = []
    class_stats = {}
    box_areas = []
    annotations_cache = {} # stem -> list of boxes
    
    max_class_id = 0

    for img_path in tqdm(img_files, desc="Checking data"):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        
        if not lbl_path.exists():
            logger.warning(f"Label missing for {img_path.name}, skipping.")
            continue
            
        # Read image dims
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Corrupt image {img_path.name}, skipping.")
            continue
        h_img, w_img = img.shape[:2]
        
        # Read label
        valid_boxes = []
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        has_error = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue

            max_class_id = max(max_class_id, cls_id)

            # Check normalized range
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                # logger.warning(f"Normalized coord out of bounds in {stem}")
                # Auto-clip in xyxy conversion
                pass
            
            # Convert to absolute
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, w_img, h_img)
            
            # Clip to image
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(0, min(x2, w_img - 1))
            y2 = max(0, min(y2, h_img - 1))
            
            # Sanity checks
            if x2 <= x1 or y2 <= y1:
                has_error = True
                continue
                
            area = (x2 - x1) * (y2 - y1)
            if area < 1.0: # Filter super small trash
                continue
                
            box_areas.append(area)
            class_stats[cls_id] = class_stats.get(cls_id, 0) + 1
            valid_boxes.append({'class_id': cls_id, 'bbox': [x1, y1, x2, y2]})

        if valid_boxes:
            valid_samples.append(str(img_path)) # Store absolute path
            annotations_cache[str(img_path)] = valid_boxes
        else:
            logger.warning(f"No valid boxes for {stem}, skipping.")
            
    # Stats
    logger.info(f"Valid samples: {len(valid_samples)} / {len(img_files)}")
    logger.info(f"Max class ID: {max_class_id}")
    logger.info(f"Total objects: {sum(class_stats.values())}")
    if box_areas:
        logger.info(f"Mean box area: {np.mean(box_areas):.2f}, Min: {np.min(box_areas)}, Max: {np.max(box_areas)}")

    # Split
    random.seed(42)
    random.shuffle(valid_samples)
    split_idx = int(len(valid_samples) * args.val_ratio)
    val_set = valid_samples[:split_idx]
    train_set = valid_samples[split_idx:]
    
    logger.info(f"Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Save processed data info
    output_data = {
        'num_classes': max_class_id + 1, # +1 because 0-indexed
        'train': train_set,
        'val': val_set,
        'annotations': annotations_cache # Cache cleaned annotations to avoid re-parsing
    }
    
    out_file = args.output
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved split and cleaned annotations to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--img_dir_name', default='images')
    parser.add_argument('--lbl_dir_name', default='labels')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--output', default='data_split.json')
    args = parser.parse_args()
    
    sanity_check(args)
