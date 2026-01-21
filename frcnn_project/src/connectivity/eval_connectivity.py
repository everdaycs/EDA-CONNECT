import os
import json
import torch
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import networkx as nx

from weak_labeling import WeakLabeler
from infer_connectivity import ConnectivityInferencer

class ConnectivityEvaluator:
    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.labeler = WeakLabeler(self.config)
        self.inferencer = ConnectivityInferencer(config_path, model_path)
        
    def _get_gt_nets(self, img_path, cpnt_path, wire_path):
        """
        Extract ground truth net clusters using the weak labeler logic
        """
        # comp2nets returns mapping {comp_idx: {net_id_1, net_id_2...}}
        comp2nets = self.labeler._map_comp_to_net(
            img_path=None, # Not used in current _map_comp_to_net if we pass H, W
            cpnt_data=json.load(open(cpnt_path)),
            wire_data=json.load(open(wire_path)),
            H=2000, W=2000 # Default/Approx, labeling logic uses it for mask rendering
        )
        # Note: I need to fix _map_comp_to_net in weak_labeling.py to handle 
        # missing image if it's just for mask size, or read img size.
        
        # Convert comp2nets into clusters of indices
        # If two comps share a net_id, they are connected.
        G = nx.Graph()
        G.add_nodes_from(range(len(json.load(open(cpnt_path)))))
        
        comp_indices = list(comp2nets.keys())
        for i in range(len(comp_indices)):
            for j in range(i + 1, len(comp_indices)):
                u, v = comp_indices[i], comp_indices[j]
                if comp2nets[u].intersection(comp2nets[v]):
                    G.add_edge(u, v)
        
        return list(nx.connected_components(G))

    def evaluate_dataset(self):
        img_dir = self.config['data']['image_dir']
        cpnt_dir = self.config['data']['cpnt_label_dir']
        wire_dir = self.config['data']['wire_label_dir']
        
        import glob
        images = sorted(glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg')))
        
        all_gt_pairs = []
        all_pred_probs = []
        
        net_metrics = [] # To store cluster-level metrics
        
        print(f"Evaluating connectivity on {len(images)} images (Using Ground Truth Components)...")
        
        for img_p in tqdm(images):
            base = os.path.splitext(os.path.basename(img_p))[0]
            c_p = os.path.join(cpnt_dir, base + "_cpnt.json")
            w_p = os.path.join(wire_dir, base + "_wire_bbox.json")
            
            if not os.path.exists(c_p) or not os.path.exists(w_p):
                continue
                
            # 1. Get GT Clusters (Using GT Comp + GT Wire)
            # Fix: map_comp_to_net needs image size
            import cv2
            img = cv2.imread(img_p)
            H, W = img.shape[:2]
            
            cpnt_data = json.load(open(c_p))
            wire_data = json.load(open(w_p))
            
            comp2nets = self.labeler._map_comp_to_net(cpnt_data, wire_data, H, W)
            
            # 2. Get Predicted Clusters (Using GT Comp + Image)
            # Our inferencer already takes cpnt_path (we pass GT path)
            pred_nets_objs = self.inferencer.predict(img_p, c_p)
            pred_clusters = [set(obj['component_ids']) for obj in pred_nets_objs]
            
            # --- Pairwise Analysis for Edge Metrics ---
            # Re-generate candidates to see how model did on specific edges
            centers = []
            for c in cpnt_data:
                bbox = c['bbox']
                centers.append([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
            
            if len(centers) < 2: continue
            
            # (Reuse candidate logic or just evaluate all pairs in GT connected components)
            # Let's evaluate purely based on "Is connected in GT cluster" vs "Is connected in Predicted cluster"
            # for all pairs considered by kNN.
            
            # For simplicity, let's use a Cluster Comparison Metric: Pairwise F1
            # Total possible pairs N*(N-1)/2
            N = len(cpnt_data)
            gt_adj = np.zeros((N, N))
            pred_adj = np.zeros((N, N))
            
            # Fill GT
            for u in range(N):
                for v in range(u+1, N):
                    if comp2nets[u].intersection(comp2nets[v]):
                        gt_adj[u, v] = 1
            
            # Fill Pred
            for cluster in pred_clusters:
                clist = list(cluster)
                for i in range(len(clist)):
                    for j in range(i+1, len(clist)):
                        u, v = clist[i], clist[j]
                        pred_adj[u, v] = 1
                        pred_adj[v, u] = 1
            
            # Flatten only the unique pairs
            iu = np.triu_indices(N, k=1)
            all_gt_pairs.extend(gt_adj[iu])
            all_pred_probs.extend(pred_adj[iu])
            
        # Compute Metrics
        y_true = np.array(all_gt_pairs)
        y_pred = np.array(all_pred_probs)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print("\n" + "="*30)
        print("CONNECTIVITY EVALUATION REPORT")
        print(f"Dataset Size: {len(images)} images")
        print(f"Component Source: GROUND TRUTH")
        print("-" * 30)
        print(f"Pairwise Accuracy:  {accuracy:.4f}")
        print(f"Pairwise Precision: {precision:.4f} (Over-connection risk)")
        print(f"Pairwise Recall:    {recall:.4f} (Missing-connection risk)")
        print(f"Pairwise F1-Score:  {f1:.4f}")
        print("="*30)

if __name__ == "__main__":
    config = "/home/kaga/Desktop/EDA-Connect/frcnn_project/configs/edge/edge_config.yaml"
    model = "/home/kaga/Desktop/EDA-Connect/frcnn_project/outputs/checkpoints_edge/best_edge_model.pth"
    
    evaluator = ConnectivityEvaluator(config, model)
    evaluator.evaluate_dataset()
