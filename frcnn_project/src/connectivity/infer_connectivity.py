import os
import json
import cv2
import torch
import numpy as np
import yaml
import networkx as nx
import argparse
from scipy.spatial import cKDTree

# Import model def
from edge_model import EdgeCNN

class ConnectivityInferencer:
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(ConnectivityInferencer.NumpyEncoder, self).default(obj)

    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = EdgeCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.img_size = self.config['model']['input_size'] # [W, H]

    def _prepare_patch(self, img, u_bbox, v_bbox):
        # Same logic as dataset
        x1 = min(u_bbox[0], v_bbox[0])
        y1 = min(u_bbox[1], v_bbox[1])
        x2 = max(u_bbox[2], v_bbox[2])
        y2 = max(u_bbox[3], v_bbox[3])
        
        w = x2 - x1
        h = y2 - y1
        diag = np.sqrt(w**2 + h**2)
        pad = max(16, int(0.1 * diag))
        
        px1 = max(0, int(x1 - pad))
        py1 = max(0, int(y1 - pad))
        px2 = min(img.shape[1], int(x2 + pad))
        py2 = min(img.shape[0], int(y2 + pad))
        
        patch = img[py1:py2, px1:px2]
        if patch.size == 0:
             return np.zeros((3, self.img_size[1], self.img_size[0]), dtype=np.float32)
             
        patch = cv2.resize(patch, tuple(self.img_size))
        # Norm
        patch = patch.astype(np.float32) / 255.0
        # Mean/Std typical imagenet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        patch = (patch - mean) / std
        
        return patch.transpose(2, 0, 1)

    def _get_geom(self, u_bbox, v_bbox):
        ucx, ucy = (u_bbox[0]+u_bbox[2])/2, (u_bbox[1]+u_bbox[3])/2
        vcx, vcy = (v_bbox[0]+v_bbox[2])/2, (v_bbox[1]+v_bbox[3])/2
        dx = vcx - ucx
        dy = vcy - ucy
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        return np.array([dx, dy, dist, angle, 
                         u_bbox[2]-u_bbox[0], u_bbox[3]-u_bbox[1], 
                         v_bbox[2]-v_bbox[0], v_bbox[3]-v_bbox[1]], dtype=np.float32)

    def predict(self, image_path, cpnt_json_path, threshold=0.5):
        img = cv2.imread(image_path)
        with open(cpnt_json_path) as f:
            cpnt_data = json.load(f)
            
        if len(cpnt_data) < 2:
            return {"nets": []}

        # 1. Generate Candidates (inference time kNN)
        centers = []
        for c in cpnt_data:
            bbox = c['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append([cx, cy])
            
        k = 12
        max_dist = 0.4 * max(img.shape[0], img.shape[1])
        tree = cKDTree(centers)
        
        candidates = []
        for i in range(len(centers)):
            dists, idxs = tree.query(centers[i], k=min(k, len(centers)))
            # cKDTree returns list if k>1
            if not isinstance(dists, np.ndarray) and not isinstance(dists, list): # single result
                 dists = [dists]
                 idxs = [idxs]

            for d, j in zip(dists, idxs):
                if i >= j: continue # Unique pairs, undirected
                if d > max_dist: continue
                candidates.append((i, j))
        
        if not candidates: return {"nets": []}

        # 2. Batch Inference
        batch_size = 64
        all_probs = []
        
        for idx in range(0, len(candidates), batch_size):
            batch_cands = candidates[idx:idx+batch_size]
            patches = []
            geoms = []
            
            for u, v in batch_cands:
                p = self._prepare_patch(img, cpnt_data[u]['bbox'], cpnt_data[v]['bbox'])
                g = self._get_geom(cpnt_data[u]['bbox'], cpnt_data[v]['bbox'])
                patches.append(p)
                geoms.append(g)
                
            p_tensor = torch.tensor(np.array(patches), dtype=torch.float32).to(self.device)
            g_tensor = torch.tensor(np.array(geoms), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(p_tensor, g_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                
        # 3. Clustering
        G = nx.Graph()
        G.add_nodes_from(range(len(cpnt_data)))
        
        for k, (u, v) in enumerate(candidates):
            if all_probs[k] > threshold:
                G.add_edge(u, v, weight=float(all_probs[k]))
                
        clusters = list(nx.connected_components(G))
        
        # Format Result
        results = []
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2: continue # Ignore singletons? or keep
            
            net_obj = {
                "net_id": f"NET_{i:03d}",
                "component_ids": list(cluster),
                "components": [cpnt_data[nid] for nid in cluster]
            }
            results.append(net_obj)
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True)
    parser.add_argument('--cpnt', required=True)
    parser.add_argument('--config', default="/home/kaga/Desktop/EDA-Connect/frcnn_project/configs/edge/edge_config.yaml")
    parser.add_argument('--model', default="/home/kaga/Desktop/EDA-Connect/frcnn_project/outputs/checkpoints_edge/best_edge_model.pth")
    args = parser.parse_args()
    
    infer = ConnectivityInferencer(args.config, args.model)
    nets = infer.predict(args.img, args.cpnt)
    
    print(json.dumps(nets, indent=2, cls=ConnectivityInferencer.NumpyEncoder))
