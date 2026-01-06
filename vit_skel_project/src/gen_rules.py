import os
import cv2
import glob
import argparse
import numpy as np
import sys
from tqdm import tqdm

# Add project root to sys.path for shared imports
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from src.skeleton_cv import extract_skeleton_cv

def generate_rule_skeleton(img_path, save_path):
    try:
        skel_img = extract_skeleton_cv(img_path)
        cv2.imwrite(save_path, skel_img)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    images = glob.glob(os.path.join(args.img_dir, "*.*"))
    images = [f for f in images if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Generating rule-based skeletons for {len(images)} images...")
    for img_p in tqdm(images):
        name = os.path.basename(img_p)
        stem = os.path.splitext(name)[0]
        out_p = os.path.join(args.out_dir, f"{stem}.png")
        generate_rule_skeleton(img_p, out_p)
        
    print("Done.")
