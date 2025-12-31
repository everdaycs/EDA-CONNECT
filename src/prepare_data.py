import os
import glob
import random
import yaml

# Configuration
base_dir = '/home/kaga/Desktop/EDA-Connect/cpnt_detect_demo'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')
output_dir = '/home/kaga/Desktop/EDA-Connect'

# Get all images
image_files = glob.glob(os.path.join(image_dir, '*.png')) + \
              glob.glob(os.path.join(image_dir, '*.jpg')) + \
              glob.glob(os.path.join(image_dir, '*.jpeg'))
image_files.sort()

# Shuffle and split
random.seed(42)
random.shuffle(image_files)

split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Write train.txt and val.txt
with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for file in train_files:
        f.write(file + '\n')

with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
    for file in val_files:
        f.write(file + '\n')

print(f"Created train.txt with {len(train_files)} images.")
print(f"Created val.txt with {len(val_files)} images.")

# Create data.yaml
# We found max class ID is 90, so we need at least 91 classes.
# We will create generic names.
nc = 91
names = [f"class_{i}" for i in range(nc)]

data_config = {
    'path': output_dir,
    'train': 'train.txt',
    'val': 'val.txt',
    'nc': nc,
    'names': names
}

with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("Created data.yaml")
