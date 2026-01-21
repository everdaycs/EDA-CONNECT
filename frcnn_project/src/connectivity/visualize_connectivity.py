import os
import json
import cv2
import torch
import numpy as np
import argparse
import random
from infer_connectivity import ConnectivityInferencer

def get_color(idx):
    """Generate a consistent random color for a given ID."""
    random.seed(idx)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def visualize(image_path, cpnt_json_path, config_path, model_path, output_path, threshold=0.5):
    # 1. Initialize Inference
    infer = ConnectivityInferencer(config_path, model_path)
    
    # 2. Get predictions
    # We want both the final nets and the raw edge predictions for a better visualization
    # Let's borrow some logic from predict but keep the intermediate graph
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    with open(cpnt_json_path) as f:
        cpnt_data = json.load(f)

    # We perform inference manually here to get edge probabilities
    # (Or we could modify ConnectivityInferencer, but keeping it simple for a standalone tool)
    nets = infer.predict(image_path, cpnt_json_path, threshold=threshold)
    
    # Create an overlay for drawing
    overlay = img.copy()
    
    # Map component index to Net ID and Color
    comp_to_net = {}
    net_colors = {}
    for net in nets:
        net_id = net['net_id']
        color = get_color(hash(net_id))
        net_colors[net_id] = color
        for comp_id in net['component_ids']:
            comp_to_net[comp_id] = net_id

    # 3. Draw connections (Edges with P > threshold)
    # Re-run candidate generation to draw lines
    centers = []
    for c in cpnt_data:
        bbox = c['bbox']
        centers.append([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
    
    # Draw Lines for connectivity
    # Note: To be perfectly accurate with the model, we'd need all_probs, 
    # but drawing lines between components in the same net is a good approximation/visualization.
    for net in nets:
        color = net_colors[net['net_id']]
        cids = net['component_ids']
        # For visualization, draw a spanning tree or lines between all pairs in a net?
        # Drawing a line between components that were predicted as connected is best.
        # But we only have clusters here. Let's just draw lines between adjacent in cluster for simplicity
        # or just draw the result of the net.
        
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                u, v = cids[i], cids[j]
                # To avoid a complete graph (too messy), only draw if they are close or were candidates
                p1 = centers[u]
                p2 = centers[v]
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if dist < 500: # Threshold for drawing lines to keep it clean
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1, cv2.LINE_AA)

    # 4. Draw Component Bboxes
    for i, cpnt in enumerate(cpnt_data):
        bbox = [int(x) for x in cpnt['bbox']]
        net_id = comp_to_net.get(i, "NC") # NC = Not Connected
        color = net_colors.get(net_id, (128, 128, 128))
        
        # Draw box
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
        
        # Label
        label = f"{net_id}" if net_id != "NC" else ""
        if label:
            cv2.putText(img, label, (bbox[0], bbox[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 5. Blend and Save
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help="Path to input circuit image")
    parser.add_argument('--cpnt', required=True, help="Path to component detection JSON")
    parser.add_argument('--config', default="frcnn_project/configs/edge/edge_config.yaml")
    parser.add_argument('--model', default="frcnn_project/outputs/checkpoints_edge/best_edge_model.pth")
    parser.add_argument('--out', default="frcnn_project/outputs/vis/connectivity_result.png")
    parser.add_argument('--thresh', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    workspace_root = "/home/kaga/Desktop/EDA-Connect"
    img_path = os.path.join(workspace_root, args.img) if not os.path.isabs(args.img) else args.img
    cpnt_path = os.path.join(workspace_root, args.cpnt) if not os.path.isabs(args.cpnt) else args.cpnt
    config_path = os.path.join(workspace_root, args.config) if not os.path.isabs(args.config) else args.config
    model_path = os.path.join(workspace_root, args.model) if not os.path.isabs(args.model) else args.model
    out_path = os.path.join(workspace_root, args.out) if not os.path.isabs(args.out) else args.out
    
    visualize(img_path, cpnt_path, config_path, model_path, out_path, threshold=args.thresh)
