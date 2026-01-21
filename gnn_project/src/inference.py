import os
import torch
import numpy as np
import json
import networkx as nx
from model import EdgePredGNN
import argparse

class LogicInference:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Model
        self.model = EdgePredGNN(in_channels=6, hidden_channels=32).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()

    def _prepare_graph(self, cpnt_data, wire_data):
        # Simplified version of GraphBuilder logic for single sample
        # 1. Nodes
        nodes = []
        idx = 0
        
        # Components
        # Assuming input format: [{'bbox': [x1,y1,x2,y2], 'class_id': int}, ...]
        for item in cpnt_data:
            bbox = item.get('bbox', [0,0,0,0])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            nodes.append({
                'id': idx, 'type': 0, 
                'pos': np.array([cx, cy]), 'bbox': np.array(bbox), 'dim': np.array([w, h])
            })
            idx += 1
            
        # Wires
        # Assuming input format: [{'points': [[x,y]...], 'bbox': [x1,y1,x2,y2]}, ...] 
        # OR the format from yolo_to_json: dictionary of NetIDs.
        # Let's support the LIST of wire dicts for inference (output of detector).
        if isinstance(wire_data, dict):
            # Convert dict format back to list if needed, or handle as is
             for net_id, segments in wire_data.items():
                for seg in segments:
                    x, y, w, h = seg['x'], seg['y'], seg['width'], seg['height']
                    cx, cy = x + w/2, y + h/2
                    nodes.append({
                        'id': idx, 'type': 1,
                        'pos': np.array([cx, cy]), 'bbox': np.array([x, y, x+w, y+h]), 'dim': np.array([w, h])
                    })
                    idx += 1
        elif isinstance(wire_data, list):
             for seg in wire_data:
                # Handle varying formats. Assuming 'bbox' exists
                bbox = seg.get('bbox', [0,0,0,0])
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                cx, cy = (bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2
                nodes.append({
                    'id': idx, 'type': 1,
                    'pos': np.array([cx, cy]), 'bbox': np.array(bbox), 'dim': np.array([w, h])
                })
                idx += 1
                
        if len(nodes) == 0: return None
        
        # 2. Features
        NORM = 2000.0
        x_feats = []
        for n in nodes:
            feats = [
                n['pos'][0]/NORM, n['pos'][1]/NORM, 
                n['dim'][0]/NORM, n['dim'][1]/NORM,
                1.0 if n['type'] == 1 else 0.0,
                1.0 if n['type'] == 0 else 0.0
            ]
            x_feats.append(feats)
        x = torch.tensor(x_feats, dtype=torch.float).to(self.device)
        
        # 3. Candidates (Edges)
        edge_indices = []
        edge_attrs = []
        raw_edges = [] # Store pairs (u, v) map back later
        
        CANDIDATE_DIST = 150.0
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n_i, n_j = nodes[i], nodes[j]
                
                # Geo Check
                dist = np.linalg.norm(n_i['pos'] - n_j['pos'])
                
                # IOU
                boxA, boxB = n_i['bbox'], n_j['bbox']
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                
                if dist < CANDIDATE_DIST or iou > 0:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    
                    attr = [dist/NORM, iou]
                    edge_attrs.append(attr)
                    edge_attrs.append(attr)
                    
                    raw_edges.append((i, j))
        
        if len(edge_indices) == 0:
            return None
            
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).to(self.device)
        
        return x, edge_index, edge_attr, nodes, raw_edges

    def predict(self, cpnt_path, wire_path, threshold=0.5):
        with open(cpnt_path) as f: cpnt_data = json.load(f)
        with open(wire_path) as f: wire_data = json.load(f)
        
        res = self._prepare_graph(cpnt_data, wire_data)
        if res is None: return {}
        
        x, edge_index, edge_attr, nodes, raw_edges = res
        
        with torch.no_grad():
            logits = self.model(x, edge_index, edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        # Reconstruct Clusters
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        
        # edge_index has shape [2, E], probs has [E, 1]
        # We need to map back to unique edges.
        # Since we added bidirectional edges, we can just iterate.
        
        rows, cols = edge_index.cpu().numpy()
        
        for k in range(len(probs)):
            u, v = rows[k], cols[k]
            p = probs[k]
            
            if p > threshold:
                G.add_edge(u, v, weight=p)
        
        # Connected Components
        clusters = list(nx.connected_components(G))
        
        # Formatting Output
        net_results = {}
        for idx, cluster in enumerate(clusters):
            # Assign a Net ID
            net_id = f"NET_{idx+1:03d}"
            
            members = []
            for node_idx in cluster:
                n = nodes[node_idx]
                members.append({
                    'type': 'component' if n['type'] == 0 else 'wire',
                    'bbox': n['bbox'].tolist(),
                    'original_index': n['id'] # Relative to the concatenated list
                })
            
            net_results[net_id] = members
            
        print(f"Found {len(clusters)} logic nets.")
        return net_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpnt_file', type=str, required=True)
    parser.add_argument('--wire_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='/home/kaga/Desktop/EDA-Connect/gnn_project/checkpoints/best_gnn.pth')
    args = parser.parse_args()
    
    inferencer = LogicInference(args.model_path)
    nets = inferencer.predict(args.cpnt_file, args.wire_file)
    
    # Save output
    base_name = os.path.basename(args.cpnt_file).replace('_cpnt.json', '')
    out_path = f'gnn_project/inference_output_{base_name}.json'
    with open(out_path, 'w') as f:
        json.dump(nets, f, indent=4)
    print(f"Saved inference result to {out_path}")
