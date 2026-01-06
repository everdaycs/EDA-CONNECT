import os
import json
import glob
from tqdm import tqdm

def txt_to_json(data_root):
    label_dir = os.path.join(data_root, 'labels')
    out_dir = os.path.join(data_root, 'det_json') # Create standard "det_json"
    os.makedirs(out_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    print(f"Converting {len(txt_files)} YOLO logs to JSON in {out_dir}...")
    
    for txt_path in tqdm(txt_files):
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        full_json_path = os.path.join(out_dir, f"{stem}.json")
        
        comps = []
        # We need image size to denormalize if we want absolute.
        # But prepare_dataset.py handles normalization itself.
        # Wait, prepare_dataset expects JSON to have [x1, y1, x2, y2].
        # It usually reads JSON. Let's see prepare_dataset again.
        
        # Actually, prepare_dataset.py ALREADY reads .txt directly now (I modified it).
        # BUT infer.py expects a JSON file with 'bbox'.
        
        # Let's verify what verify what infer.py expects.
        # infer.py: with open(args.det_json, 'r') as f: comps = json.load(f)
        # It expects comps to be a list of dicts.
        
        # Since we don't know the image size here (unless we open the image), 
        # we have a problem. YOLO txt is normalized.
        # infer.py loads the image, so it knows H, W.
        # BUT infer.py takes the JSON as input.
        # So the JSON *must* contain absolute coordinates?
        
        # Let's look at `infer.py`:
        # for c in comps: ... b = c['bbox'] ... cx = (b[0]+b[2])/2/W ..
        # So infer.py expects absolute coordinates [x1, y1, x2, y2].
        
        # So we MUST open the image to convert.
        image_path = os.path.join(data_root, 'images', f"{stem}.png")
        if not os.path.exists(image_path):
             image_path = os.path.join(data_root, 'images', f"{stem}.jpg")
        
        if not os.path.exists(image_path): continue
        
        import cv2
        img = cv2.imread(image_path)
        if img is None: continue
        H, W = img.shape[:2]
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cid = int(parts[0])
                ncx, ncy, nw, nh = map(float, parts[1:5])
                
                cx, cy = ncx * W, ncy * H
                w, h = nw * W, nh * H
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                
                comps.append({
                    'class_id': cid,
                    'bbox': [x1, y1, x2, y2],
                    'conf': 1.0 # variable if from real inference
                })
        
        with open(full_json_path, 'w') as f:
            json.dump(comps, f, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python convert_yolo_to_json.py <data_root>")
        exit(1)
    txt_to_json(sys.argv[1])
