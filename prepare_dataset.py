"""
Dataset Preparation Script for Arabic Sign Language (ArSL21L)
Organizes the dataset into YOLOv5 format with train/val split.

Expected input structure (after extracting the Kaggle download):
    data/raw/  (place extracted dataset here)
        ├── images/
        │   ├── img1.jpg
        │   └── ...
        └── labels/
            ├── img1.txt
            └── ...

Output structure:
    data/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
"""

import os
import shutil
import random
import glob
import yaml

# Configuration
RAW_DATA_DIR = os.path.join("data", "raw")
OUTPUT_DIR = "data"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# 32 Arabic Sign Language letter classes
ARABIC_CLASSES = [
    "ain", "al", "aleff", "bb", "dal", "dha", "dhad", "fa",
    "gaaf", "ghain", "ha", "haa", "jeem", "kaaf", "khaa", "la",
    "laam", "meem", "nun", "ra", "saad", "seen", "sheen", "ta",
    "taa", "thaa", "thal", "toot", "waw", "ya", "yaa", "zay"
]


def find_dataset_structure(raw_dir):
    """Auto-detect dataset structure - handles various folder layouts."""
    images_dir = None
    labels_dir = None

    # Check common structures
    candidates = [
        (os.path.join(raw_dir, "images"), os.path.join(raw_dir, "labels")),
        (os.path.join(raw_dir, "train", "images"), os.path.join(raw_dir, "train", "labels")),
        (raw_dir, raw_dir),  # flat structure
    ]

    for img_candidate, lbl_candidate in candidates:
        if os.path.isdir(img_candidate):
            images_dir = img_candidate
            labels_dir = lbl_candidate
            break

    # If not found, search recursively for image files
    if images_dir is None:
        for root, dirs, files in os.walk(raw_dir):
            img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if len(img_files) > 10:
                images_dir = root
                # Look for labels sibling
                parent = os.path.dirname(root)
                potential_labels = os.path.join(parent, "labels")
                if os.path.isdir(potential_labels):
                    labels_dir = potential_labels
                else:
                    labels_dir = root  # labels might be in same folder
                break

    return images_dir, labels_dir


def prepare_yolo_dataset():
    """Organize dataset into YOLOv5 train/val format."""
    print("=" * 60)
    print("Arabic Sign Language Dataset Preparation")
    print("=" * 60)

    # Check raw data exists
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"\nERROR: Raw data directory not found: {RAW_DATA_DIR}")
        print("\nPlease download the dataset first:")
        print("  1. pip install kaggle")
        print("  2. kaggle datasets download -d ammarsayedtaha/arabic-sign-language-dataset-2022")
        print(f"  3. Extract the contents into: {RAW_DATA_DIR}/")
        return False

    # Find dataset structure
    images_dir, labels_dir = find_dataset_structure(RAW_DATA_DIR)
    if images_dir is None:
        print(f"\nERROR: Could not find images in {RAW_DATA_DIR}")
        print("Please ensure the dataset is extracted correctly.")
        return False

    print(f"\nFound images in: {images_dir}")
    print(f"Found labels in: {labels_dir}")

    # Collect image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))

    image_files = list(set(image_files))  # Remove duplicates
    print(f"Total images found: {len(image_files)}")

    if len(image_files) == 0:
        print("ERROR: No images found!")
        return False

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images")

    # Copy files
    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        print(f"\nCopying {split_name} files...")
        for img_path in file_list:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]

            # Copy image
            dst_img = os.path.join(OUTPUT_DIR, "images", split_name, img_name)
            shutil.copy2(img_path, dst_img)

            # Find and copy corresponding label
            label_name = base_name + ".txt"
            label_path = os.path.join(labels_dir, label_name)

            # Try finding label in subdirectories too
            if not os.path.exists(label_path):
                label_candidates = glob.glob(
                    os.path.join(labels_dir, "**", label_name), recursive=True
                )
                if label_candidates:
                    label_path = label_candidates[0]

            if os.path.exists(label_path):
                dst_label = os.path.join(OUTPUT_DIR, "labels", split_name, label_name)
                shutil.copy2(label_path, dst_label)

    # Detect actual class names from labels
    detected_classes = set()
    for label_file in glob.glob(os.path.join(OUTPUT_DIR, "labels", "train", "*.txt")):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    detected_classes.add(int(parts[0]))

    num_classes = max(detected_classes) + 1 if detected_classes else len(ARABIC_CLASSES)
    print(f"\nDetected {num_classes} classes in labels")

    # Generate data.yaml
    data_yaml = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': num_classes,
        'names': ARABIC_CLASSES[:num_classes]
    }

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    # Also save a copy at project root for easy access
    shutil.copy2(yaml_path, "data.yaml")

    print(f"\ndata.yaml saved to: {yaml_path}")
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nNext step: Train the model")
    print("  - Google Colab: Upload train_colab.ipynb")
    print("  - Local: python train.py")

    return True


if __name__ == "__main__":
    prepare_yolo_dataset()
