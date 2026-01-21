
import os
import shutil
import json
from pathlib import Path

def extract_intersecting_data(
    line_seg_root: str,
    cpnt_detect_root: str,
    output_root: str
):
    """
    Extracts data that exists in both datasets (intersection) and saves it to a new directory.
    Prioritizes JSON labels for logic training.
    """
    
    # Define source paths
    line_images_dir = Path(line_seg_root) / "images"
    line_labels_dir = Path(line_seg_root) / "labels_json" # Use the original JSONs
    
    cpnt_images_dir = Path(cpnt_detect_root) / "images"
    cpnt_labels_dir = Path(cpnt_detect_root) / "det_json" 
    
    # Define destination paths
    out_path = Path(output_root)
    out_images_dir = out_path / "images"
    out_line_labels_dir = out_path / "line_labels"
    out_cpnt_labels_dir = out_path / "cpnt_labels"
    
    # Create directories
    for p in [out_images_dir, out_line_labels_dir, out_cpnt_labels_dir]:
        p.mkdir(parents=True, exist_ok=True)
        
    # Get file sets (stems)
    print(f"Scanning {line_images_dir}...")
    line_files = list(line_images_dir.glob("*.png"))
    
    print(f"Scanning {cpnt_images_dir}...")
    cpnt_files = {f.name: f for f in cpnt_images_dir.glob("*.png")} # Map filename to path
    
    intersect_pairs = []
    
    for l_file in line_files:
        # Try to match mapping logic:
        # Check specific prefix removal
        if l_file.name.startswith("scenario_35_"):
            potential_cpnt_name = l_file.name.replace("scenario_35_", "")
            if potential_cpnt_name in cpnt_files:
                intersect_pairs.append((l_file, cpnt_files[potential_cpnt_name]))
                continue
        
        # Direct match check (just in case)
        if l_file.name in cpnt_files:
             intersect_pairs.append((l_file, cpnt_files[l_file.name]))

    print(f"Found {len(intersect_pairs)} intersecting pairs.")
    
    # Process files
    for src_img_line, src_img_cpnt in intersect_pairs:
        stem = src_img_line.stem # Use line seg stem as the unified ID
        
        # 1. Copy Image (Use line seg one as base)
        dst_img = out_images_dir / f"{stem}.png"
        shutil.copy2(src_img_line, dst_img)
        
        # 2. Copy Line Label (JSON)
        # Pattern: [stem]_wire_bbox.json
        line_json_name = f"{stem}_wire_bbox.json"
        src_line_json = line_labels_dir / line_json_name
        dst_line_json = out_line_labels_dir / line_json_name
        
        if src_line_json.exists():
            shutil.copy2(src_line_json, dst_line_json)
        else:
            print(f"Warning: Missing line label for {stem}")

        # 3. Copy Component Label (JSON)
        # Cpnt stem might differ, use the file we found
        cpnt_stem = src_img_cpnt.stem
        
        # Try to find corresponding json in cpnt_detect_root/det_json
        # Is it [cpnt_stem].json? or [cpnt_stem]_det.json?
        # Let's search with glob using the short stem
        potential_cpnt_jsons = list(cpnt_labels_dir.glob(f"{cpnt_stem}*.json"))
        
        if potential_cpnt_jsons:
             src_cpnt_json = potential_cpnt_jsons[0] # Take first match
             # Save with the UNIFIED stem (scenario_35_...) to keep matches clear
             dst_cpnt_json = out_cpnt_labels_dir / f"{stem}_cpnt.json" 
             shutil.copy2(src_cpnt_json, dst_cpnt_json)
        else:
            print(f"Warning: Missing component label for {cpnt_stem} (Unified: {stem})")

    print(f"Extraction complete. Data saved to {output_root}")

if __name__ == "__main__":
    extract_intersecting_data(
        line_seg_root="/home/kaga/Desktop/EDA-Connect/line_seg_demo_20260114",
        cpnt_detect_root="/home/kaga/Desktop/EDA-Connect/cpnt_detect_demo",
        output_root="/home/kaga/Desktop/EDA-Connect/joint_training_data"
    )
