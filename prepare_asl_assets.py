"""
Prepare real ASL hand sign images from the ayuraj/asl-dataset Kaggle dataset.

Usage:
    1. kaggle datasets download -d ayuraj/asl-dataset --path data/asl_raw2
    2. python prepare_asl_assets.py
"""

import os
import zipfile
from PIL import Image

RAW_ZIP = os.path.join("data", "asl_raw2", "asl-dataset.zip")
EXTRACT_DIR = os.path.join("data", "asl_raw2", "extracted")
TRAIN_DIR = os.path.join(EXTRACT_DIR, "asl_dataset")
ASL_SIGNS_DIR = os.path.join("assets", "asl_signs")

# Dataset uses lowercase folder names (a-z); map to uppercase output filenames
LETTER_MAP = {chr(i): chr(i - 32) for i in range(ord('a'), ord('z') + 1)}


def main():
    # Step 1: Unzip
    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting {RAW_ZIP} ...")
        with zipfile.ZipFile(RAW_ZIP, 'r') as z:
            z.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    else:
        print("Archive already extracted, skipping unzip.")

    os.makedirs(ASL_SIGNS_DIR, exist_ok=True)

    # Step 2: Pick one image per letter, resize, save
    success = 0
    for folder_name, dest_name in LETTER_MAP.items():
        folder_path = os.path.join(TRAIN_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: folder not found: {folder_path}")
            continue

        images = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not images:
            print(f"  WARNING: no images in {folder_path}")
            continue

        src_path = os.path.join(folder_path, images[0])
        img = Image.open(src_path).convert("RGB")
        img = img.resize((200, 200), Image.LANCZOS)

        dest_path = os.path.join(ASL_SIGNS_DIR, f"{dest_name}.png")
        img.save(dest_path)
        print(f"  {folder_name} -> {dest_name}.png")
        success += 1

    print(f"\nDone -- {success} ASL sign images saved to {ASL_SIGNS_DIR}/")
    print("Note: SPACE.png kept as-is (no SPACE folder in this dataset).")


if __name__ == "__main__":
    main()
