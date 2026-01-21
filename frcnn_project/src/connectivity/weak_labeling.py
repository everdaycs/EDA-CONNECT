import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree

class WeakLabeler:
    def __init__(self, config):
        self.config = config
        self.r_out = config['data']['weak_label']['border_expand_out']
        self.r_in = config['data']['weak_label']['border_expand_in']
        self.k_kernel = config['data']['weak_label']['wire_dilate_kernel']
        self.t_contact = config['data']['weak_label']['contact_threshold']

    def generate_labels(self, img_path, cpnt_path, wire_path):
        """
        Generate (i, j, label) triplets for graph edges
        """
        # Load Data
        img = cv2.imread(img_path)
        if img is None: return None
        H, W = img.shape[:2]
        
        with open(cpnt_path) as f: cpnt_data = json.load(f)
        with open(wire_path) as f: wire_data = json.load(f) # Dictionary of nets
        
        # 1. Map Components to Net IDs (Weak Supervision)
        comp2nets = self._map_comp_to_net(cpnt_data, wire_data, H, W)
        
        # 2. Build Candidate Edges
        edges = self._build_candidate_edges(cpnt_data, H, W)
        
        # 3. Label Edges
        labeled_edges = []
        for u, v in edges:
            nets_u = comp2nets.get(u, set())
            nets_v = comp2nets.get(v, set())
            
            # Intersection implies connection
            common = nets_u.intersection(nets_v)
            label = 1 if len(common) > 0 else 0
            
            labeled_edges.append({
                'u': u, 'v': v, 'label': label,
                'u_cls': cpnt_data[u].get('class_id', 0),
                'v_cls': cpnt_data[v].get('class_id', 0),
                'u_bbox': cpnt_data[u]['bbox'],
                'v_bbox': cpnt_data[v]['bbox']
            })
            
        return labeled_edges

    def _map_comp_to_net(self, cpnt_data, wire_data, H, W):
        """
        Determine which nets touch which component
        """
        comp2nets = {} # {comp_idx: {net_id1, ...}}
        
        # Pre-render wires to masks per NetID to save time? 
        # Or iterate Nets -> Wires -> Check overlap with all Comp Rings
        
        # Create Component Rings
        comp_rings = []
        for idx, comp in enumerate(cpnt_data):
            bbox = comp['bbox'] # [x1, y1, x2, y2]
            
            # Create Ring Mask
            mask = np.zeros((H, W), dtype=np.uint8)
            
            # Outer
            x1 = max(0, int(bbox[0]) - self.r_out)
            y1 = max(0, int(bbox[1]) - self.r_out)
            x2 = min(W, int(bbox[2]) + self.r_out)
            y2 = min(H, int(bbox[3]) + self.r_out)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            # Inner (subtract)
            x1_in = max(0, int(bbox[0]) - self.r_in)
            y1_in = max(0, int(bbox[1]) - self.r_in)
            x2_in = min(W, int(bbox[2]) + self.r_in)
            y2_in = min(H, int(bbox[3]) + self.r_in)
            cv2.rectangle(mask, (x1_in, y1_in), (x2_in, y2_in), 0, -1)
            
            comp_rings.append(mask)
            comp2nets[idx] = set()

        # Check Intersection with Wires
        # Handle dict or list format for wire_data
        if isinstance(wire_data, list):
             # Try to group by label or if raw bbox lists
             # Assuming structure from previous step: dict {net_id: [segments]}
             # If it's pure list from detection, we can't do net Logic training without NetID annotations.
             # The prompt says: "NetID导线bbox" in training data is Key.
             # So we assume wire_path points to GT data which has NetIDs.
             # Our joint file `line_labels` has JSON dicts: "net_id": [list of bboxes]
             pass
        
        for net_id, segments in wire_data.items():
            # Render Net mask
            net_mask = np.zeros((H, W), dtype=np.uint8)
            for seg in segments:
                # support both xywh and xyxy or point lists
                if 'points' in seg:
                     pts = np.array(seg['points'], dtype=np.int32)
                     cv2.polylines(net_mask, [pts], False, 255, 1) # Thin line initially
                else:
                    # Box format
                    x = int(seg.get('x', 0))
                    y = int(seg.get('y', 0))
                    w = int(seg.get('width', 0))
                    h = int(seg.get('height', 0))
                    cv2.rectangle(net_mask, (x, y), (x+w, y+h), 255, -1)
            
            # Dilate Net Mask
            kernel = np.ones((self.k_kernel, self.k_kernel), np.uint8)
            net_mask = cv2.dilate(net_mask, kernel, iterations=1)
            
            # Check overlap with all comps
            for idx, ring_mask in enumerate(comp_rings):
                overlap = cv2.bitwise_and(ring_mask, net_mask)
                contact_area = cv2.countNonZero(overlap)
                
                if contact_area >= self.t_contact:
                    comp2nets[idx].add(net_id)
                    
        return comp2nets

    def _build_candidate_edges(self, cpnt_data, H, W):
        centers = []
        for c in cpnt_data:
            bbox = c['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append([cx, cy])
        
        centers = np.array(centers)
        if len(centers) < 2: return []
        
        k = min(self.config['data']['edge_gen']['k_neighbors'], len(centers)-1)
        max_dist = self.config['data']['edge_gen']['max_radius'] * max(H, W)
        
        tree = cKDTree(centers)
        
        edges = set()
        for i in range(len(centers)):
            dists, idxs = tree.query(centers[i], k=k+1) # +1 includes self
            
            for d, j in zip(dists, idxs):
                if i == j: continue
                if d > max_dist: continue
                
                # Undirected edge -> sort indices
                edge = tuple(sorted((i, j)))
                edges.add(edge)
                
        return list(edges)

if __name__ == "__main__":
    # Test Block
    pass
