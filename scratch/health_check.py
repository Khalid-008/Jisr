import torch
import cv2
import mediapipe as mp
import numpy as np
import sys

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")
print(f"MediaPipe solutions found: {hasattr(mp, 'solutions')}")
if hasattr(mp, 'solutions'):
    print(f"MediaPipe hands available: {hasattr(mp.solutions, 'hands')}")

try:
    from ultralytics import YOLO
    print("Ultralytics YOLO imported successfully")
except Exception as e:
    print(f"Ultralytics import failed: {e}")
