import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

class CircuitExtractor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.components = []
        self.wires = []
        self.graph = nx.Graph()

    def detect_components(self, image_path):
        """
        Detect components using YOLO model.
        """
        results = self.model(image_path)
        self.components = []
        
        # Process results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                self.components.append({
                    'bbox': r, # [x1, y1, x2, y2]
                    'class': cls,
                    'conf': conf,
                    'label': f"C{cls}" # Generic label
                })
        return self.components

    def extract_wires(self, image_path):
        """
        Extract wires by skeletonizing the entire image.
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Thresholding
        # Adaptive thresholding is good for varying lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. Noise removal and Closing
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Closing to bridge gaps
        closing_kernel = np.ones((5,5), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)
        
        # Dilate to thicken lines and merge breaks
        dilation_kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(closing, dilation_kernel, iterations=1)
        
        # 4. Skeletonization
        # scikit-image skeletonize expects boolean array where True is foreground
        skeleton = skeletonize(dilated > 0)
        self.skeleton_img = (skeleton * 255).astype(np.uint8)
        
        return self.skeleton_img

    def build_graph(self):
        """
        Build a graph from skeleton and components.
        """
        skeleton = self.skeleton_img // 255
        
        # Get coordinates of all skeleton pixels
        y_idxs, x_idxs = np.where(skeleton > 0)
        
        # Build a pixel graph first
        pixel_graph = nx.Graph()
        
        for y, x in zip(y_idxs, x_idxs):
            pixel_graph.add_node((x, y))
            # Check 8-neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                        if skeleton[ny, nx_]:
                            pixel_graph.add_edge((x, y), (nx_, ny))
                            
        self.graph = nx.Graph()
        
        # Add component nodes
        for i, comp in enumerate(self.components):
            self.graph.add_node(f"Comp_{i}", type='component', **comp)
            
        # Map ALL pixel nodes to components if they are inside the bbox
        node_to_comp = {}
        for node in pixel_graph.nodes():
            nx_, ny = node
            for i, comp in enumerate(self.components):
                x1, y1, x2, y2 = comp['bbox']
                # Check if node is inside or very close to bbox
                pad = 20
                if (x1 - pad <= nx_ <= x2 + pad) and (y1 - pad <= ny <= y2 + pad):
                    node_to_comp[node] = i
                    break # Assume one component per node (no overlap)
        
        print(f"Pixel graph has {pixel_graph.number_of_nodes()} nodes.")
        print(f"Mapped {len(node_to_comp)} nodes to components.")

        # Find connected components in pixel graph
        pixel_subgraphs = [pixel_graph.subgraph(c).copy() for c in nx.connected_components(pixel_graph)]
        print(f"Found {len(pixel_subgraphs)} connected components (potential nets).")
        
        # Merge close nets
        net_graph = nx.Graph()
        for i in range(len(pixel_subgraphs)):
            net_graph.add_node(i)
            
        # Collect endpoints for each net
        net_endpoints = {}
        for i, subg in enumerate(pixel_subgraphs):
            # Endpoints are nodes with degree 1 (or 0 if single pixel)
            eps = [n for n, d in subg.degree() if d == 1]
            if not eps and subg.number_of_nodes() == 1:
                eps = list(subg.nodes())
            net_endpoints[i] = eps
            
        # Check distance between nets
        # This is O(N^2) which is slow, but N is small (240)
        for i in range(len(pixel_subgraphs)):
            for j in range(i + 1, len(pixel_subgraphs)):
                # Check if any endpoint of i is close to any endpoint of j
                connected = False
                for ep_i in net_endpoints[i]:
                    for ep_j in net_endpoints[j]:
                        dist = np.sqrt((ep_i[0]-ep_j[0])**2 + (ep_i[1]-ep_j[1])**2)
                        if dist < 40: # Merge threshold increased
                            net_graph.add_edge(i, j)
                            connected = True
                            break
                    if connected: break
        
        # Find super-nets
        super_nets = list(nx.connected_components(net_graph))
        print(f"Merged into {len(super_nets)} super-nets.")
        
        connections = []
        
        for sn_idx, super_net_indices in enumerate(super_nets):
            # Find all components touched by this super-net
            connected_comps = set()
            for net_idx in super_net_indices:
                subg = pixel_subgraphs[net_idx]
                for node in subg.nodes():
                    if node in node_to_comp:
                        connected_comps.add(node_to_comp[node])
            
            connected_comps = list(connected_comps)
            
            if len(connected_comps) > 1:
                # Add edges between all components in this net
                net_name = f"Net_{sn_idx}"
                self.graph.add_node(net_name, type='net')
                
                for comp_idx in connected_comps:
                    comp_name = f"Comp_{comp_idx}"
                    self.graph.add_edge(comp_name, net_name)
                    connections.append((comp_name, net_name))

        return connections

    def save_netlist(self, output_path):
        with open(output_path, 'w') as f:
            f.write("Netlist\n")
            f.write("=======\n")
            
            # List components
            f.write("\nComponents:\n")
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'component':
                    f.write(f"{node}: Class {data['class']} at {data['bbox']}\n")
            
            # List Connections
            f.write("\nConnections:\n")
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'net':
                    neighbors = list(self.graph.neighbors(node))
                    comps = [n for n in neighbors if self.graph.nodes[n].get('type') == 'component']
                    if len(comps) > 1:
                        f.write(f"{node}: " + " -- ".join(comps) + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python circuit_extractor.py <image_path>")
        return

    image_path = sys.argv[1]
    model_path = '/home/kaga/Desktop/EDA-Connect/runs/detect/train_demo/weights/best.pt'
    
    extractor = CircuitExtractor(model_path)
    
    print("Detecting components...")
    extractor.detect_components(image_path)
    print(f"Found {len(extractor.components)} components.")
    
    print("Extracting wires...")
    extractor.extract_wires(image_path)
    
    print("Building graph...")
    extractor.build_graph()
    
    output_txt = "output_netlist.txt"
    extractor.save_netlist(output_txt)
    print(f"Netlist saved to {output_txt}")
    
    # Visualization (Optional, save to file)
    img = cv2.imread(image_path)
    for comp in extractor.components:
        x1, y1, x2, y2 = comp['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"C{comp['class']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # Draw skeleton
    skeleton_bgr = cv2.cvtColor(extractor.skeleton_img, cv2.COLOR_GRAY2BGR)
    skeleton_bgr[extractor.skeleton_img > 0] = [0, 0, 255] # Red skeleton
    
    # Overlay
    added_image = cv2.addWeighted(img, 0.7, skeleton_bgr, 0.3, 0)
    cv2.imwrite("result_visualization.png", added_image)
    print("Visualization saved to result_visualization.png")

if __name__ == '__main__':
    main()
