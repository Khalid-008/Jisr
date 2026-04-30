"""
Prepare ASL Alphabet Dataset for YOLO Detection Training.

Converts the classification-style ASL alphabet dataset (images in folders, no labels)
into YOLO detection format by auto-generating bounding box labels using MediaPipe Hands.

Input:  data/asl_raw/extracted/asl_alphabet_train/asl_alphabet_train/{A-Z,space}/
Output: data/asl_yolo/train/images/, data/asl_yolo/train/labels/,
        data/asl_yolo/valid/images/, data/asl_yolo/valid/labels/
        + data_asl.yaml

Usage:
    python prepare_asl_dataset.py
    python prepare_asl_dataset.py --skip-existing   # resume interrupted run
    python prepare_asl_dataset.py --workers 8        # more parallel workers
"""

import argparse
import os
import random
import shutil
from multiprocessing import Pool, cpu_count

import cv2
import mediapipe as mp
import yaml


# Classes to use (skip 'del' and 'nothing' — meta-gestures, not letters)
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "SPACE"
]

# Map folder names to class index
FOLDER_TO_CLASS = {}
for i, name in enumerate(CLASS_NAMES):
    folder = name.lower() if name != "SPACE" else "space"
    FOLDER_TO_CLASS[folder] = i

# Default paths
DEFAULT_INPUT = os.path.join("data", "asl_raw", "extracted", "asl_alphabet_train", "asl_alphabet_train")
DEFAULT_OUTPUT = os.path.join("data", "asl_yolo")

# Fallback bounding box when MediaPipe fails (images are tightly cropped)
FALLBACK_BBOX = "0.5 0.5 0.8 0.8"


def process_image(args):
    """Process a single image: detect hand with MediaPipe and write YOLO label.

    Args:
        args: tuple of (image_path, label_path, class_id, skip_existing)

    Returns:
        tuple of (success: bool, used_fallback: bool)
    """
    image_path, label_path, class_id, skip_existing = args

    if skip_existing and os.path.exists(label_path):
        return True, False

    img = cv2.imread(image_path)
    if img is None:
        return False, False

    h, w = img.shape[:2]

    # Run MediaPipe Hands
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    ) as hands:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

    used_fallback = False

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add 15% margin
        margin_x = (x_max - x_min) * 0.15
        margin_y = (y_max - y_min) * 0.15
        x_min = max(0.0, x_min - margin_x)
        y_min = max(0.0, y_min - margin_y)
        x_max = min(1.0, x_max + margin_x)
        y_max = min(1.0, y_max + margin_y)

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        bw = x_max - x_min
        bh = y_max - y_min

        label_line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    else:
        label_line = f"{class_id} {FALLBACK_BBOX}"
        used_fallback = True

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        f.write(label_line + "\n")

    return True, used_fallback


def main():
    parser = argparse.ArgumentParser(description="Prepare ASL dataset for YOLO training")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to ASL classification dataset")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output directory for YOLO dataset")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images that already have labels")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()), help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    print("=" * 60)
    print("ASL Alphabet → YOLO Detection Dataset Preparation")
    print("=" * 60)

    if not os.path.isdir(args.input):
        print(f"\nERROR: Input directory not found: {args.input}")
        print("Download the dataset first: kaggle datasets download -d grassknoted/asl-alphabet")
        return

    # Collect all images
    print(f"\nScanning {args.input}...")
    all_items = []  # list of (image_path, class_id, folder_name)

    for folder_name, class_id in FOLDER_TO_CLASS.items():
        folder_path = os.path.join(args.input, folder_name if folder_name != "space" else "space")
        # Try case variations
        if not os.path.isdir(folder_path):
            folder_path = os.path.join(args.input, folder_name.upper())
        if not os.path.isdir(folder_path):
            folder_path = os.path.join(args.input, folder_name.capitalize())
        if not os.path.isdir(folder_path):
            print(f"  WARNING: Folder not found for class '{CLASS_NAMES[class_id]}', skipping")
            continue

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        for img_name in images:
            all_items.append((os.path.join(folder_path, img_name), class_id, CLASS_NAMES[class_id]))

    print(f"  Found {len(all_items)} images across {len(FOLDER_TO_CLASS)} classes")

    if not all_items:
        print("ERROR: No images found!")
        return

    # Generate labels with MediaPipe (parallelized)
    print(f"\nGenerating YOLO bounding box labels (workers={args.workers})...")
    print("  This may take 30-60 minutes for 87K images...")

    temp_label_dir = os.path.join(args.output, "_temp_labels")
    os.makedirs(temp_label_dir, exist_ok=True)

    tasks = []
    for img_path, class_id, class_name in all_items:
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(temp_label_dir, f"{class_name}_{img_basename}.txt")
        tasks.append((img_path, label_path, class_id, args.skip_existing))

    stats = {name: {"total": 0, "success": 0, "fallback": 0} for name in CLASS_NAMES}

    with Pool(args.workers) as pool:
        results = pool.map(process_image, tasks)

    for (img_path, class_id, class_name), (success, used_fallback) in zip(all_items, results):
        stats[class_name]["total"] += 1
        if success:
            stats[class_name]["success"] += 1
        if used_fallback:
            stats[class_name]["fallback"] += 1

    # Print stats
    print("\n  Per-class statistics:")
    print(f"  {'Class':<8} {'Total':>6} {'Success':>8} {'Fallback':>9}")
    print("  " + "-" * 35)
    total_fallback = 0
    for name in CLASS_NAMES:
        s = stats[name]
        total_fallback += s["fallback"]
        print(f"  {name:<8} {s['total']:>6} {s['success']:>8} {s['fallback']:>9}")
    print(f"\n  Total fallbacks: {total_fallback}/{len(all_items)}")

    # Train/val split
    print(f"\nSplitting into train/val (80/20, seed={args.seed})...")
    random.seed(args.seed)

    # Group by class for stratified split
    class_items = {name: [] for name in CLASS_NAMES}
    for (img_path, class_id, class_name), (success, _) in zip(all_items, results):
        if success:
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(temp_label_dir, f"{class_name}_{img_basename}.txt")
            class_items[class_name].append((img_path, label_path))

    train_dir_img = os.path.join(args.output, "train", "images")
    train_dir_lbl = os.path.join(args.output, "train", "labels")
    val_dir_img = os.path.join(args.output, "valid", "images")
    val_dir_lbl = os.path.join(args.output, "valid", "labels")

    for d in [train_dir_img, train_dir_lbl, val_dir_img, val_dir_lbl]:
        os.makedirs(d, exist_ok=True)

    train_count = 0
    val_count = 0

    for class_name, items in class_items.items():
        random.shuffle(items)
        split_idx = int(len(items) * 0.8)
        train_items = items[:split_idx]
        val_items = items[split_idx:]

        for img_path, label_path in train_items:
            basename = f"{class_name}_{os.path.basename(img_path)}"
            lbl_basename = os.path.splitext(basename)[0] + ".txt"
            shutil.copy2(img_path, os.path.join(train_dir_img, basename))
            shutil.copy2(label_path, os.path.join(train_dir_lbl, lbl_basename))
            train_count += 1

        for img_path, label_path in val_items:
            basename = f"{class_name}_{os.path.basename(img_path)}"
            lbl_basename = os.path.splitext(basename)[0] + ".txt"
            shutil.copy2(img_path, os.path.join(val_dir_img, basename))
            shutil.copy2(label_path, os.path.join(val_dir_lbl, lbl_basename))
            val_count += 1

    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")

    # Generate data_asl.yaml
    yaml_path = "data_asl.yaml"
    yaml_data = {
        "train": os.path.join(args.output, "train", "images"),
        "val": os.path.join(args.output, "valid", "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  Dataset config saved to: {yaml_path}")

    # Cleanup temp labels
    shutil.rmtree(temp_label_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"Upload {args.output}/ to Google Drive for Colab training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
