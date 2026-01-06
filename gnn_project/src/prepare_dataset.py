import os
import json
import cv2
import numpy as np
import torch
import glob
from skimage.morphology import skeletonize
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor

def process_single_sample(args_tuple):
    """
    Worker function for parallel processing.
    """
    name, data_root, output_dir, r_graph, d_attach, num_classes = args_tuple
    
    image_dir = os.path.join(data_root, 'images')
    label_dir = os.path.join(data_root, 'labels')
    
    # Paths
    img_path = os.path.join(image_dir, f"{name}.png")
    if not os.path.exists(img_path):
         img_path = os.path.join(image_dir, f"{name}.jpg")
    
    lbl_path = os.path.join(label_dir, f"{name}.txt")
    
    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        return None

    # Check if output already exists (Resume capability)
    save_path = os.path.join(output_dir, f"{name}.pt")
    if os.path.exists(save_path):
        return save_path

    # 1. Load Image
    img = cv2.imread(img_path)
    if img is None: return None
    H, W = img.shape[:2]

    # 2. Skeletonize (Image Processing)
    # This is often the bottleneck, so mp helps a lot
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    skel_bool = skeletonize(thresh > 0)
    skel_thinned = skel_bool.astype(np.uint8) * 255
    
    # Connected Components
    num_labels_cc, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(skel_thinned, connectivity=8)

    # 3. Load Components
    comps = []
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            comps.append({'class_id': cid, 'bbox': [x1, y1, x2, y2]})

    num_nodes = len(comps)
    # If too few nodes, graph training might be unstable or useless, but let's keep even 1 or 2
    if num_nodes == 0: return None

    # 4. Attach Components to Nets (Vectorized-ish via Mask ROI)
    # Instead of iterating pixels, iterate components and check region in label map
    comp_net_map = {i: set() for i in range(num_nodes)}
    
    for i, c in enumerate(comps):
        b = c['bbox']
        x1 = int(max(0, b[0] - d_attach))
        y1 = int(max(0, b[1] - d_attach))
        x2 = int(min(W, b[2] + d_attach))
        y2 = int(min(H, b[3] + d_attach))
        
        if x2 <= x1 or y2 <= y1: continue
        
        # Crop the labels map
        roi_labels = labels_cc[y1:y2, x1:x2]
        unique_nets = np.unique(roi_labels) # This is fast
        
        for nid in unique_nets:
            if nid != 0: # 0 is background
                comp_net_map[i].add(nid)

    # 5. Generate Positive Pairs (Ground Truth for Link Prediction)
    pos_pairs = set()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Intersection of sets
            if not comp_net_map[i].isdisjoint(comp_net_map[j]):
                pos_pairs.add((i, j))
                pos_pairs.add((j, i))

    # 6. Build Graph Data
    node_features = []
    node_pos = []
    
    for c in comps:
        b = c['bbox']
        # Normalized features [class, cx, cy, w, h]
        cx = (b[0] + b[2]) / 2.0 / W
        cy = (b[1] + b[3]) / 2.0 / H
        w = (b[2] - b[0]) / W
        h = (b[3] - b[1]) / H
        
        node_features.append([c['class_id'], cx, cy, w, h])
        node_pos.append([cx * W, cy * H]) # Abs pos for distance

    x = torch.tensor(node_features, dtype=torch.float)
    pos_abs = torch.tensor(node_pos, dtype=torch.float)
    
    # Vectorized Radius Graph Generation
    # cdist is much faster than nested loops
    dist_matrix = torch.cdist(pos_abs, pos_abs) # [N, N]
    
    # Mask for radius graph
    mask = dist_matrix < r_graph
    # Remove self-loops
    mask.fill_diagonal_(False)
    
    # Get indices
    row, col = torch.where(mask)
    edge_index = torch.stack([row, col], dim=0)
    
    # Generate edge labels
    # We have edge_index [2, E], want y [E]
    y = []
    for k in range(edge_index.size(1)):
        u, v = edge_index[0, k].item(), edge_index[1, k].item()
        l = 1.0 if (u, v) in pos_pairs else 0.0
        y.append(l)
        
    y = torch.tensor(y, dtype=torch.float)
    
    # Save normalized pos for visualization/debug if needed
    normalized_pos = x[:, 1:3]

    data = Data(x=x, edge_index=edge_index, y=y, pos=normalized_pos)
    torch.save(data, save_path)
    return save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./gnn_project/processed_data')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--r_graph', type=int, default=300)
    parser.add_argument('--d_attach', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_dir = os.path.join(args.data_root, 'images')
    if not os.path.exists(image_dir):
        print("Image dir not found")
        exit(1)
        
    samples = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"🚀 Processing {len(samples)} samples with {args.num_workers} workers...")
    
    # Prepare args for map
    tasks = [(s, args.data_root, args.output_dir, args.r_graph, args.d_attach, args.num_classes) for s in samples]
    
    processed_count = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for _ in tqdm(executor.map(process_single_sample, tasks), total=len(tasks)):
            if _ is not None:
                processed_count += 1
        
    print(f"✅ Successfully processed {processed_count} graphs.")
