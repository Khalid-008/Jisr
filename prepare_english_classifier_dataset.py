"""
Local helper to organize the Kaggle ASL alphabet download into the layout
expected by train_english_classifier_colab.ipynb.

Usage:
    python prepare_english_classifier_dataset.py --src /path/to/asl_alphabet_train/asl_alphabet_train
                                                  --dst datasets/asl_cropped

MediaPipe crops each image the same way inference will, closing the train/test gap.
Images where MediaPipe finds no hand are skipped (except the NOTHING class).
"""

import argparse
import os
import random
import shutil

import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except AttributeError:
    raise SystemExit("ERROR: mediapipe.solutions not available. Install mediapipe==0.10.14")

FOLDER_MAP = {
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E", "f": "F",
    "g": "G", "h": "H", "i": "I", "j": "J", "k": "K", "l": "L",
    "m": "M", "n": "N", "o": "O", "p": "P", "q": "Q", "r": "R",
    "s": "S", "t": "T", "u": "U", "v": "V", "w": "W", "x": "X",
    "y": "Y", "z": "Z", "space": "SPACE", "del": "DEL", "nothing": "NOTHING",
}


def crop_hand_square(img_bgr, hands_ctx):
    h, w = img_bgr.shape[:2]
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = hands_ctx.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lms = result.multi_hand_landmarks[0]
    xs  = [lm.x for lm in lms.landmark]
    ys  = [lm.y for lm in lms.landmark]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    x0 = max(0, int((min(xs) - bw * 0.2) * w))
    y0 = max(0, int((min(ys) - bh * 0.2) * h))
    x1 = min(w, int((max(xs) + bw * 0.2) * w))
    y1 = min(h, int((max(ys) + bh * 0.2) * h))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    side = max(crop.shape[:2])
    pad  = np.zeros((side, side, 3), dtype=np.uint8)
    oh, ow = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    pad[oh:oh + crop.shape[0], ow:ow + crop.shape[1]] = crop
    return pad


def process(src_root, dst_root, seed=42):
    random.seed(seed)
    total_ok = total_skip = 0

    for folder_name, class_name in FOLDER_MAP.items():
        src = os.path.join(src_root, folder_name)
        if not os.path.isdir(src):
            print(f"  WARNING: not found: {src}")
            continue

        images = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        random.shuffle(images)
        n = len(images)
        splits = {
            "train": images[: int(n * 0.8)],
            "val":   images[int(n * 0.8): int(n * 0.9)],
            "test":  images[int(n * 0.9):],
        }

        cls_ok = cls_skip = 0
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                            min_detection_confidence=0.3) as hands:
            for split, img_list in splits.items():
                out_dir = os.path.join(dst_root, split, class_name)
                os.makedirs(out_dir, exist_ok=True)
                for fname in img_list:
                    img = cv2.imread(os.path.join(src, fname))
                    if img is None:
                        continue
                    cropped = crop_hand_square(img, hands)
                    if cropped is None:
                        if class_name == "NOTHING":
                            h, w = img.shape[:2]
                            side  = min(h, w)
                            cropped = cv2.cvtColor(img[:side, :side], cv2.COLOR_BGR2RGB)
                        else:
                            cls_skip += 1
                            continue
                    cv2.imwrite(os.path.join(out_dir, fname), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                    cls_ok += 1

        total_ok   += cls_ok
        total_skip += cls_skip
        print(f"  {class_name}: {cls_ok} saved, {cls_skip} skipped")

    print(f"\nDone. {total_ok} images saved, {total_skip} skipped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Path to asl_alphabet_train/asl_alphabet_train/")
    parser.add_argument("--dst", default="datasets/asl_cropped", help="Output directory")
    args = parser.parse_args()
    process(args.src, args.dst)


if __name__ == "__main__":
    main()
