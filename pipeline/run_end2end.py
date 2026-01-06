import os
import argparse
import subprocess
import shutil
import cv2
import json

def run_command(cmd, desc):
    print(f"[*] {desc}...")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error in {desc}: {e}")
        exit(1)

def run_end2end(args):
    img_path = os.path.abspath(args.image)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(args.out_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Detection / Labels Handling
    det_json_path = args.det_json
    
    if args.label and os.path.exists(args.label):
        print(f"[*] Using Ground Truth labels: {args.label}")
        temp_json = os.path.join(out_dir, "gt_det.json")
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Error: Could not read image {img_path}")
            exit(1)
        H, W = img.shape[:2]
        
        comps = []
        with open(args.label, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = int((cx - w/2) * W)
                y1 = int((cy - h/2) * H)
                x2 = int((cx + w/2) * W)
                y2 = int((cy + h/2) * H)
                comps.append({'class_id': cid, 'bbox': [x1, y1, x2, y2], 'conf': 1.0})
        
        with open(temp_json, 'w') as f:
            json.dump(comps, f)
        det_json_path = temp_json
    
    if not det_json_path or not os.path.exists(det_json_path):
        print("[!] No detection JSON or Label provided. YOLO inference not yet integrated in this script.")
        exit(1)
    
    # 2. Stage 1: DeiT Skeleton Inference
    skel_out_path = os.path.join(out_dir, "skeleton_deit.png")
    # Use sys.executable to ensure we use the same python interpreter
    import sys
    py_exec = sys.executable
    cmd_deit = f"{py_exec} vit_skel_project/src/infer.py --img_path '{img_path}' --ckpt '{args.deit_ckpt}' --out_mask '{skel_out_path}'"
    run_command(cmd_deit, "Stage 1: DeiT Skeleton Inference")
    
    # 3. Stage 2: GNN Inference
    # Note: GNN infer usually outputs to a dir. We want it in out_dir
    # GNN infer currently takes --out_dir and creates {stem}_netlist.txt inside
    cmd_gnn = f"{py_exec} gnn_project/src/infer.py --image '{img_path}' --det_json '{det_json_path}' --ckpt '{args.gnn_ckpt}' --out_dir '{out_dir}' --skel_source deit"
    run_command(cmd_gnn, "Stage 2: GNN Netlist Extraction")
    
    # 4. Final Cleanup / organizing
    # infer.py already generates overlay and netlist.
    print(f"[+] Pipeline Completed. Results in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hardcoded defaults for quick execution
    DEFAULT_IMG = "cpnt_detect_demo/images/003_01_001.png"
    DEFAULT_LBL = "cpnt_detect_demo/labels/003_01_001.txt"
    DEFAULT_DEIT = "vit_skel_project/checkpoints/best.pt"
    DEFAULT_GNN = "gnn_project/checkpoints/best_model.pth"

    parser.add_argument('--image', type=str, default=DEFAULT_IMG, help="Input schematic image")
    parser.add_argument('--label', type=str, default=DEFAULT_LBL, help="YOLO format ground truth label (.txt)")
    parser.add_argument('--det_json', type=str, help="Detection JSON file (if labels not provided)")
    parser.add_argument('--deit_ckpt', type=str, default=DEFAULT_DEIT, help="Path to DeiT Stage 1 checkpoint")
    parser.add_argument('--gnn_ckpt', type=str, default=DEFAULT_GNN, help="Path to GNN Stage 2 checkpoint")
    parser.add_argument('--out_dir', type=str, default='output', help="Results output directory")
    args = parser.parse_args()
    
    # Check if files exist before running
    for path_attr in ['image', 'label', 'deit_ckpt', 'gnn_ckpt']:
        path = getattr(args, path_attr)
        if path and not os.path.exists(path):
            print(f"[!] Warning: {path_attr} path does not exist: {path}")

    run_end2end(args)
