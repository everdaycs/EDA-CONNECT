import json
import os
import glob
from PIL import Image
from tqdm import tqdm
import random

def convert_to_yolo():
    # Setup paths
    ROOT_DIR = "/home/kaga/Desktop/EDA-Connect/line_seg_demo_20260114"
    IMAGES_DIR = os.path.join(ROOT_DIR, "images")
    LABELS_JSON_DIR = os.path.join(ROOT_DIR, "labels")
    
    # Create YOLO labels directory
    YOLO_LABELS_DIR = os.path.join(ROOT_DIR, "labels_yolo")
    os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
    
    # Get all images
    image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    
    # Data splitting lists
    train_files = []
    val_files = []
    
    print(f"Found {len(image_paths)} images. Starting conversion...")
    
    for img_path in tqdm(image_paths):
        # Get basics
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Construct corresponding JSON path (e.g., scenario..._wire_bbox.json)
        json_path = os.path.join(LABELS_JSON_DIR, f"{name_no_ext}_wire_bbox.json")
        
        if not os.path.exists(json_path):
            # print(f"Warning: No label found for {basename}, skipping...")
            # If training on background images is needed, create empty txt file.
            # But here let's assume we want valid samples.
            continue
            
        # Get image dimensions for normalization
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        # Prepare YOLO label content
        yolo_lines = []
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Iterate through all nets (keys) in the json
            for net_id, segments in data.items():
                for seg in segments:
                    # JSON format: x, y, width, height (assuming x,y is Top-Left based on typical json exports)
                    x, y, w, h = seg['x'], seg['y'], seg['width'], seg['height']
                    
                    # Convert to Center-X, Center-Y, Width, Height
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    
                    # Normalize (0.0 - 1.0)
                    cx /= img_w
                    cy /= img_h
                    w /= img_w
                    h /= img_h
                    
                    # Clamp values to be safe
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))
                    
                    # Class ID 0 for 'wire'
                    yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_path}")
            continue
            
        # Write YOLO label file (.txt)
        txt_filename = f"{name_no_ext}.txt"
        txt_path = os.path.join(YOLO_LABELS_DIR, txt_filename)
        
        with open(txt_path, 'w') as f:
            f.write("\n".join(yolo_lines))
            
        # Add to split list
        if random.random() < 0.1: # 10% validation
            val_files.append(img_path)
        else:
            train_files.append(img_path)
            
    # Save split files
    with open(os.path.join(ROOT_DIR, "train_wire.txt"), "w") as f:
        f.write("\n".join(train_files))
        
    with open(os.path.join(ROOT_DIR, "val_wire.txt"), "w") as f:
        f.write("\n".join(val_files))
        
    print(f"Conversion Done.")
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print(f"YOLO labels saved to: {YOLO_LABELS_DIR}")

if __name__ == "__main__":
    convert_to_yolo()
