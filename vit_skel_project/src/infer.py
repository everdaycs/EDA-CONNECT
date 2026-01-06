import os
import argparse
import torch
import cv2
import numpy as np
from model import ViTSkeleton

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ViTSkeleton(model_name='deit_tiny_patch16_224', img_size=224).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    img_orig = cv2.imread(args.img_path)
    if img_orig is None:
        raise ValueError("Image not found")
        
    H, W = img_orig.shape[:2]
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Resize for model (must match training size usually for PosEmbed to be happy, 
    # though timm dynamic_img_size handles some flex)
    target_size = 224 # Or argument
    img_resized = cv2.resize(img, (target_size, target_size))
    
    inp = img_resized.astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(inp) # Output is likely target_size x target_size
        out = torch.sigmoid(out).squeeze().cpu().numpy()
        
    # Resize back to original
    out_full = cv2.resize(out, (W, H))
    mask = (out_full > 0.5).astype(np.uint8) * 255
    
    os.makedirs(os.path.dirname(args.out_mask), exist_ok=True)
    cv2.imwrite(args.out_mask, mask)
    print(f"Saved skeleton mask to {args.out_mask}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_mask', type=str, required=True)
    args = parser.parse_args()
    
    infer(args)
