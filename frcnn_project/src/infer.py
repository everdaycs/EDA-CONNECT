import os
import torch
import cv2
import argparse
import yaml
import json
import glob
from tqdm import tqdm
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.utils.general import get_device

def infer(args):
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = get_device(args.device if args.device else cfg['train']['device'])
    
    # Load Model structure
    # We need num_classes. If training produced a json, use it, else default 91
    # Check if split file exists to get Num classes, or input arg
    num_classes = 91
    split_file = cfg['data'].get('split_file')
    if os.path.exists(split_file):
        with open(split_file) as f:
             meta = json.load(f)
             num_classes = meta['num_classes'] + 1
             
    model = get_model(num_classes, cfg)
    
    # Load Weights
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Handle state dict structure (if saved with 'model' key)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()
    
    # Images
    if os.path.isdir(args.input):
        img_files = glob.glob(os.path.join(args.input, "*.png")) + \
                    glob.glob(os.path.join(args.input, "*.jpg"))
    else:
        img_files = [args.input]
        
    results = []
    
    # Tiling settings
    use_tiling = cfg['inference']['tiling']['enabled']
    tile_size = cfg['inference']['tiling']['tile_size']
    overlap = cfg['inference']['tiling']['overlap']
    
    for img_path in tqdm(img_files, desc="Inferencing"):
        img_original = cv2.imread(img_path)
        if img_original is None: continue
        
        # Simple Logic: Resize or Full
        # For this demo, let's just do full image inference (torchvision handles size)
        img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            prediction = model([img_tensor])[0]
            
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # Filter by score
        thresh = args.conf_thresh
        keep = scores >= thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Save JSON Format
        base_name = os.path.basename(img_path)
        for b, s, l in zip(boxes, scores, labels):
            results.append({
                "file_name": base_name,
                "class_id": int(l) - 1, # Revert to 0-based
                "bbox": b.tolist(), # xyxy
                "conf": float(s)
            })
            
        # Visualization
        if args.vis_dir:
            os.makedirs(args.vis_dir, exist_ok=True)
            vis_img = img_original.copy()
            for b, l, s in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"{l-1}:{s:.2f}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(args.vis_dir, base_name), vis_img)
            
    # Save Results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Image file or dir")
    parser.add_argument('--checkpoint', required=True, help="Path to .pth")
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output', default='results.json')
    parser.add_argument('--vis_dir', default=None)
    parser.add_argument('--conf_thresh', type=float, default=0.3) # Low thresh for recall
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    
    infer(args)
