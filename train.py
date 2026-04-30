"""
YOLOv5 Training Script for Arabic Sign Language Detection
Can be run locally or adapted for Google Colab.

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 8  # lighter config
"""

import argparse
import os
import torch
from ultralytics import YOLO


def train(args):
    print("=" * 60)
    print("Arabic Sign Language - YOLOv5 Training")
    print("=" * 60)

    # Check data.yaml exists
    if not os.path.exists(args.data):
        print(f"\nERROR: {args.data} not found!")
        print("Run prepare_dataset.py first to prepare the dataset.")
        return

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow (6-12 hours).")
        print("Consider using Google Colab with train_colab.ipynb instead.")

    # Load YOLOv5s model (small, fast)
    print(f"\nLoading YOLOv5s model...")
    model = YOLO("yolov5s.pt")

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_size}")
    print(f"  Data: {args.data}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device,
        patience=20,  # early stopping
        save=True,
        project="runs",
        name="arabic_sign",
        exist_ok=True,
    )

    # Copy best weights to models/
    best_pt = os.path.join("runs", "arabic_sign", "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs("models", exist_ok=True)
        dest = os.path.join("models", "arabic_sign_best.pt")
        import shutil
        shutil.copy2(best_pt, dest)
        print(f"\nBest weights saved to: {dest}")

        # Export to ONNX
        print("\nExporting to ONNX...")
        best_model = YOLO(dest)
        best_model.export(format="onnx")
        print(f"ONNX model saved to: models/arabic_sign_best.onnx")
    else:
        print(f"\nWARNING: Best weights not found at {best_pt}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv5 for Arabic Sign Language")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    args = parser.parse_args()
    train(args)
