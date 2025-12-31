import os
import glob

label_dir = '/home/kaga/Desktop/EDA-Connect/cpnt_detect_demo/labels'
label_files = glob.glob(os.path.join(label_dir, '*.txt'))

classes = set()

for file in label_files:
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                classes.add(int(parts[0]))

print(f"Found classes: {sorted(list(classes))}")
print(f"Number of classes: {len(classes)}")
