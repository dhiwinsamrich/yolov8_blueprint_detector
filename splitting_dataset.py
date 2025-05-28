import os
import random
from shutil import copyfile

def split_dataset(image_dir, label_dir, train_ratio=0.8):
    all_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # Create directories if they don't exist
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/train/labels', exist_ok=True)
    os.makedirs('data/val/images', exist_ok=True)
    os.makedirs('data/val/labels', exist_ok=True)
    
    # Copy files to train and val directories
    for file in train_files:
        base_name = os.path.splitext(file)[0]
        copyfile(f"{image_dir}/{file}", f"data/train/images/{file}")
        copyfile(f"{label_dir}/{base_name}.txt", f"data/train/labels/{base_name}.txt")
    
    for file in val_files:
        base_name = os.path.splitext(file)[0]
        copyfile(f"{image_dir}/{file}", f"data/val/images/{file}")
        copyfile(f"{label_dir}/{base_name}.txt", f"data/val/labels/{base_name}.txt")

split_dataset('data/images', 'data/labels')