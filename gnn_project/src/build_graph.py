import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import math

class GraphBuilder:
    def __init__(self, data_root, save_dir):
        self.data_root = data_root
        self.save_dir = save_dir
        self.image_dir = os.path.join(data_root, 'images')
        self.line_label_dir = os.path.join(data_root, 'line_labels')
        self.cpnt_label_dir = os.path.join(data_root, 'cpnt_labels')
        
        self.processed_dir = os.path.join(save_dir, 'raw_graphs')
        os.makedirs(self.processed_dir, exist_ok=True)

    def _get_files(self):
        line_files = set(f.replace('_wire_bbox.json', '') for f in os.listdir(self.line_label_dir) if f.endswith('_wire_bbox.json'))
        cpnt_files = set(f.replace('_cpnt.json', '') for f in os.listdir(self.cpnt_label_dir) if f.endswith('_cpnt.json'))
        
        valid_bases = list(line_files.intersection(cpnt_files))
        valid_bases.sort()
        print(f"Found {len(valid_bases)} valid paired samples.")
        return valid_bases

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _get_cpnt_nodes(self, cpnt_data, img_w=1000, img_h=1000):
        nodes = []
        for i, item in enumerate(cpnt_data):
            bbox = item.get('bbox', [0,0,0,0])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            nodes.append({
                'id': i,
                'type': 0, # 0 for component
                'pos': np.array([cx, cy]),
                'bbox':  np.array(bbox),
                'dim': np.array([w, h]),
                'class_id': item.get('class_id', 0),
                'net_id': None # Unknown for components
            })
        return nodes

    def _get_wire_nodes(self, wire_data, img_w=1000, img_h=1000):
        nodes = []
        idx_counter = 0
        for net_id, segments in wire_data.items():
            for seg in segments:
                x = seg['x']
                y = seg['y']
                w = seg['width']
                h = seg['height']
                
                # BBox in x,y,w,h (top-left) or center?
                # Usually LabelMe/YOLO export formats vary. 
                # Checking sample: "x": 1080.0, "y": 555.0, "width": 10.0, "height": 2
                # This looks like Top-Left X, Y, W, H.
                
                cx = x + w / 2.0
                cy = y + h / 2.0
                bbox = np.array([x, y, x+w, y+h])
                
                nodes.append({
                    'id': idx_counter,
                    'type': 1, # 1 for wire
                    'pos': np.array([cx, cy]),
                    'bbox': bbox,
                    'dim': np.array([w, h]),
                    'class_id': -1, # generic wire
                    'net_id': net_id
                })
                idx_counter += 1
        return nodes

    def _compute_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou, interArea

    def _check_geometric_connection(self, node_a, node_b, dist_thresh=15.0):
        # 1. Dist check
        dist = np.linalg.norm(node_a['pos'] - node_b['pos'])
        
        # 2. Intersection check
        iou, inter_area = self._compute_iou(node_a['bbox'], node_b['bbox'])
        
        is_close = dist < dist_thresh or inter_area > 0
        return is_close, dist, iou

    def process_all(self):
        bases = self._get_files()
        
        processed_count = 0
        for base in tqdm(bases):
            # Load
            l_path = os.path.join(self.line_label_dir, base + '_wire_bbox.json')
            c_path = os.path.join(self.cpnt_label_dir, base + '_cpnt.json')
            
            wire_data = self._load_json(l_path)
            cpnt_data = self._load_json(c_path)
            
            # TODO: Get Image size if needed for normalization. Assuming standard 2000x2000 or similar
            # For now using raw coords or normalized logic if implemented.
            
            cpnt_nodes = self._get_cpnt_nodes(cpnt_data)
            wire_nodes = self._get_wire_nodes(wire_data)
            
            # Offset wire IDs to follow component IDs
            c_len = len(cpnt_nodes)
            for wn in wire_nodes:
                wn['id'] += c_len
                
            all_nodes = cpnt_nodes + wire_nodes
            
            if len(all_nodes) == 0:
                continue

            # --- Build Features ---
            # Feat: [x_norm, y_norm, w_norm, h_norm, is_wire, is_comp]
            # Normalizing by 2000.0 (approx image size)
            NORM = 2000.0
            
            x_feats = []
            for n in all_nodes:
                feats = [
                    n['pos'][0]/NORM, 
                    n['pos'][1]/NORM, 
                    n['dim'][0]/NORM, 
                    n['dim'][1]/NORM,
                    1.0 if n['type'] == 1 else 0.0, # Is Wire
                    1.0 if n['type'] == 0 else 0.0  # Is Comp
                ]
                x_feats.append(feats)
            
            x_tensor = torch.tensor(x_feats, dtype=torch.float)

            # --- Build Edges ---
            edge_indices = []
            edge_labels = []
            edge_attrs = []
            
            # Naive full connectivity is too much N^2.
            # Radius graph strategy: connect if dist < Threshold
            # Threshold needs to be generous enough to capture real connections but prune far ones.
            # Wire-Wire connection needs overlap.
            
            CANDIDATE_DIST = 150.0 # Pixel distance to consider as candidate edge
            
            for i in range(len(all_nodes)):
                for j in range(i + 1, len(all_nodes)):
                    n_i = all_nodes[i]
                    n_j = all_nodes[j]
                    
                    is_candidate, dist, iou = self._check_geometric_connection(n_i, n_j, dist_thresh=CANDIDATE_DIST)
                    
                    if is_candidate:
                        # Determine Ground Truth Label
                        label = 0.0
                        
                        # Case 1: Wire - Wire
                        if n_i['type'] == 1 and n_j['type'] == 1:
                            if n_i['net_id'] == n_j['net_id']:
                                # Same Net ID -> Connected
                                label = 1.0
                                # But wait, if they don't touch, are they connected in the graph?
                                # Ideally yes, transitively. But for "Link Prediction" on geometric graph, 
                                # usually we predict "Direct Edge".
                                # If two wires are same NetID but far apart, they are connected via other wires.
                                # Predicting a link between them might be wrong if they don't touch directly.
                                # Refinement: Only label 1 if they strictly TOUCH/OVERLAP or are extremely close.
                                if iou > 0 or dist < 10.0:
                                     label = 1.0
                                else:
                                     # Same Net, but not physically connecting HERE.
                                     # For a graph *construction* task, an edge exists if geometry allows.
                                     # If we put an edge here, and label it 0, the GNN learns "Same Net ID wires shouldn't bridge gaps"?
                                     # Let's stick to: Label 1 if (SameNetID AND PhysicallyClose).
                                     pass
                        
                        # Case 2: Comp - Wire
                        elif n_i['type'] != n_j['type']:
                            # Using geometric intersection as GT for Comp-Wire
                            if iou > 0 or dist < 5.0:
                                label = 1.0
                        
                        # Case 3: Comp - Comp
                        # Usually don't connect directly unless abutting. 
                        # Assume 0 unless we have info.
                        pass
                        
                        # Add undirected edge
                        edge_indices.append([i, j])
                        edge_indices.append([j, i])
                        edge_labels.append(label)
                        edge_labels.append(label)
                        
                        # Edge Attr: [dist_norm, iou]
                        attr = [dist/NORM, iou]
                        edge_attrs.append(attr)
                        edge_attrs.append(attr)

            if len(edge_indices) == 0:
                # Add self loops or empty
                edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
                edge_label_tensor = torch.empty((0), dtype=torch.float)
                edge_attr_tensor = torch.empty((0, 2), dtype=torch.float)
            else:
                edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_label_tensor = torch.tensor(edge_labels, dtype=torch.float)
                edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)
            
            data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=edge_label_tensor)
            torch.save(data, os.path.join(self.processed_dir, f'{base}.pt'))
            processed_count += 1
            
        print(f"Processed {processed_count} graphs.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/kaga/Desktop/EDA-Connect/joint_training_data')
    parser.add_argument('--save_dir', default='/home/kaga/Desktop/EDA-Connect/gnn_project/processed_data')
    args = parser.parse_args()
    
    builder = GraphBuilder(args.data_root, args.save_dir)
    builder.process_all()
