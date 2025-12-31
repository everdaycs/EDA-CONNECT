# Circuit Recognition and Netlist Extraction Prototype

This project implements a prototype for recognizing electronic components and extracting connection netlists from schematic images.

## Structure

- `src/train_yolo.py`: Script to train the YOLOv8 model on the provided dataset.
- `src/circuit_extractor.py`: Main script to detect components, extract wires, and generate a netlist.
- `src/prepare_data.py`: Helper script to prepare the dataset (split train/val, create data.yaml).
- `src/scan_classes.py`: Helper script to identify class IDs.

## Usage

### 1. Setup Environment
Install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to train YOLOv8 on your dataset:
```bash
python src/train_yolo.py
```
The best model will be saved at `runs/detect/train_demo/weights/best.pt`.

### 3. Run Extraction
Run the extractor on an image:
```bash
python src/circuit_extractor.py <path_to_image>
```
Example:
```bash
python src/circuit_extractor.py cpnt_detect_demo/images/001_01_001.png
```

### 4. Output
The script generates:
- `output_netlist.txt`: A text file listing detected components and their connections.
- `result_visualization.png`: An image showing the detected components and extracted wire skeleton.

## Methodology

1.  **Component Detection**: Uses YOLOv8 to detect component bounding boxes and classes.
2.  **Wire Extraction**:
    *   Converts image to grayscale.
    *   Applies adaptive thresholding to isolate lines.
    *   Performs morphological closing and dilation to repair broken lines.
    *   Skeletonizes the binary image to 1-pixel wide paths.
3.  **Graph Construction**:
    *   Builds a graph where nodes are skeleton pixels.
    *   Maps pixel nodes to components if they fall within the component bounding box (with padding).
    *   Identifies connected components (nets) in the pixel graph.
    *   Merges spatially close nets to handle fragmentation.
    *   Outputs connections between components that share the same net.
