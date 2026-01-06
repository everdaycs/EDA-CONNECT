import os
import json
import torch
import cv2
import numpy as np
import argparse
import networkx as nx
from torch_geometric.data import Data
from models import NetlistGNN

def infer_single_image(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    # Assuming Det JSON format from prepare_dataset
    with open(args.det_json, 'r') as f:
        comps = json.load(f)
        
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError("Image not found")
    H, W = img.shape[:2]

    # Pre-calculate Edge Probs (Net Construction) IF we want more robust inference
    # But this script is GNN-only inference. The prompt says: "use skeleton + det_json generate pseudo connection".
    # Wait, infer.py IS the GNN inference. It takes nodes and builds graph. 
    # BUT, the prompt implies "Skeleton + YOLO det -> GNN connection".
    # In standard GNN inference (Link Prediction), we don't strictly NEED the skeleton mask IF the model is purely geometric (bbox distance).
    # HOWEVER, if the model was trained with skeleton-guided labels, inference is just predicting links on Fully Connected (or Radius) candidate edges.
    
    # Wait, does the USER want GNN to take the skeleton image as INPUT feature?
    # Prompt: "Stage 2... use skeleton + YOLO det generate pseudo connection/graph data, train & infer GNN"
    # This usually means Skeleton is used for LABELS (Training) or Graph Construction (Heuristic).
    # If GNN is just link prediction on nodes, it doesn't necessarily see the skeleton pixels during inference unless we add skeleton features to edge/node.
    # The current `models.py` only takes geometric features (x, y, w, h). 
    # So skeleton is ONLY used for Pseudo-Labels during training!
    # AND potentially during inference to filter candidate edges if we want a hybrid approach. 
    # BUT the prompt asks to specificy `skel_source`. 
    # If `skel_source` is used in `infer.py`, it implies we might use it to prune edges OR just for visualization overlay?
    # Re-reading: "4.1 Pseudo Label Generation... 4.2 GNN Training and Inference". 
    # The GNN model itself (SAGE/GAT) only sees the graph.
    # SO `skel_source` in inference might just be for 1) Visual Overlay 2) Optional Heuristic Pruning (if implemented).
    # Let's keep it simple: `infer.py` builds the graph based on radius (like training) and predicts probabilities. 
    # The skeleton is purely for training labels.
    # Wait, "use skeleton ... to generate pseudo connection". This is usually training.
    # BUT "infer GNN... output netlist".
    
    # Let's stick to the current logic: GNN predicts links based on geometry learned from skeleton-based labels.
    # But I will add skel_source argument to `infer.py` anyway, incase we want to overlay it.
    
    # 2. Build Graph (Inference Mode - No Pseudolabels needed)
    node_features = []
    
    for c in comps:
        cls = c['class_id']
        b = c['bbox']
        cx = (b[0] + b[2]) / 2.0 / W
        cy = (b[1] + b[3]) / 2.0 / H
        w = (b[2] - b[0]) / W
        h = (b[3] - b[1]) / H
        node_features.append([cls, cx, cy, w, h])
        
    x = torch.tensor(node_features, dtype=torch.float).to(device)
    num_nodes = len(comps)
    
    # Build Candidate Edges (SAME Logic as Training)
    # Using simple radius or fully connected if K is small
    edge_index = []
    
    # Reconstruct positions for distance check
    pos = x[:, 1:3] * torch.tensor([W, H]).to(device) # approx restore
    
    # Naive pairwise
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # Undirected
            dist = torch.norm(pos[i] - pos[j])
            
            if dist < args.r_graph:
                edge_index.append([i, j])
                edge_index.append([j, i]) # Bi-directional for GNN
                
    if len(edge_index) == 0:
        print("No edges found nearby.")
        return

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    # 3. Model Inference
    model = NetlistGNN(num_classes=args.num_classes, model_type=args.model_type).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()
    
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits)
        
    # 4. Filter & Build Netlist
    # Use NetworkX to find connected components from predicted edges
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Iterate edges (take only one direction i < j to avoid double counting if undirected logic used)
    row, col = edge_index
    for k in range(row.size(0)):
        u, v = row[k].item(), col[k].item()
        p = probs[k].item()
        
        if u < v and p > args.p_thr:
            G.add_edge(u, v, weight=p)
            
    # Connected Components -> Nets
    nets = list(nx.connected_components(G))
    
    # 5. Export
    os.makedirs(args.out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    txt_path = os.path.join(args.out_dir, f"{basename}_netlist.txt")
    vis_path = os.path.join(args.out_dir, f"{basename}_overlay.png")
    
    with open(txt_path, 'w') as f:
        f.write("Components:\n")
        for i, c in enumerate(comps):
            f.write(f"Comp_{i}: class={c['class_id']} bbox={c['bbox']}\n")
            
        f.write("\nNets:\n")
        net_idx = 0
        for comp_set in nets:
            if len(comp_set) < 2: continue # Ignore single floating variants usually
            comp_list = [f"Comp_{i}" for i in comp_set]
            f.write(f"Net_{net_idx}: {' -- '.join(comp_list)}\n")
            net_idx += 1
            
    print(f"Netlist saved to {txt_path}")
    
    # 6. Visualization
    # Draw connections
    for u, v in G.edges():
        c1 = comps[u]['bbox']
        c2 = comps[v]['bbox']
        
        pt1 = (int((c1[0]+c1[2])/2), int((c1[1]+c1[3])/2))
        pt2 = (int((c2[0]+c2[2])/2), int((c2[1]+c2[3])/2))
        
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        
    # Draw boxes
    for i, c in enumerate(comps):
        pk = c['bbox']
        cv2.rectangle(img, (int(pk[0]), int(pk[1])), (int(pk[2]), int(pk[3])), (0, 255, 0), 2)
        cv2.putText(img, str(i), (int(pk[0]), int(pk[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imwrite(vis_path, img)
    print(f"Visualization saved to {vis_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--det_json', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--model_type', type=str, default='SAGE')
    parser.add_argument('--r_graph', type=float, default=300.0) # Match prepare_dataset
    parser.add_argument('--p_thr', type=float, default=0.5)
    parser.add_argument('--skel_source', type=str, default='rule', help="For visualization or future hybrid methods")
    args = parser.parse_args()
    
    infer_single_image(args)
