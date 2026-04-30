"""
YOLOv5 Training Script for ASL (English) Sign Language Detection
Can be run locally or adapted for Google Colab.

Usage:
    python train_english.py
    python train_english.py --epochs 50 --batch-size 8  # lighter config
"""

import argparse
import os
import torch
from ultralytics import YOLO


def train(args):
    print("=" * 60)
    print("ASL (English) Sign Language - YOLOv5 Training")
    print("=" * 60)

    # Check data_asl.yaml exists
    if not os.path.exists(args.data):
        print(f"\nERROR: {args.data} not found!")
        print("Run prepare_asl_dataset.py first to prepare the dataset.")
        return

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow.")
        print("Consider using Google Colab with train_english_colab.ipynb instead.")

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
        patience=15,
        save=True,
        save_period=5,
        project="runs",
        name="english_sign",
        exist_ok=True,
    )

    # Copy best weights to models/
    best_pt = os.path.join("runs", "english_sign", "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs("models", exist_ok=True)
        dest = os.path.join("models", "english_sign_best.pt")
        import shutil
        shutil.copy2(best_pt, dest)
        print(f"\nBest weights saved to: {dest}")

        # Export to ONNX
        print("\nExporting to ONNX...")
        best_model = YOLO(dest)
        best_model.export(format="onnx")
        print(f"ONNX model saved to: models/english_sign_best.onnx")
    else:
        print(f"\nWARNING: Best weights not found at {best_pt}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv5 for ASL Sign Language")
    parser.add_argument("--data", type=str, default="data_asl.yaml", help="Path to data_asl.yaml")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    args = parser.parse_args()
    train(args)
