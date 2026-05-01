# Jisr

Sign Language Translation System — supports Arabic and English (ASL) sign language detection using YOLOv5.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## ASL (English) Detection — Known Issues & Fixes

The `grassknoted/asl-alphabet` Kaggle dataset is notoriously unrealistic: one person, one room, near-identical lighting, plain background, and the hand fills almost the entire frame in every image. Models trained on it routinely hit 99% validation mAP but collapse on real webcam input.

### Fixes Applied

The following changes address the domain mismatch / overfitting problem:

| # | Fix | Files Changed | Impact |
|---|-----|---------------|--------|
| 1 | **Match inference imgsz to training** (320→640) | `detector.py` | Restores small-object detection; biggest single-line win |
| 2 | **Per-language confidence threshold** (0.45 for English) | `config.py`, `detector.py` | Prevents overly-aggressive filtering on ASL model |
| 3 | **Drop fallback bounding boxes** — skip poor-quality images | `prepare_asl_dataset.py`, `train_english_colab.ipynb` | Stops teaching model "hand = whole frame" |
| 4 | **Strong augmentation** (mosaic, scale jitter, HSV, mixup) | `train_english_colab.ipynb`, `train_english.py` | Adds diversity the dataset lacks |

### Training

To retrain the ASL model with the fixes above:

**Google Colab (recommended):** Open `train_english_colab.ipynb` and run all cells.

**Local:**
```bash
python prepare_asl_dataset.py
python train_english.py
```

### Testing Conditions

Until the dataset is replaced with a more diverse one, optimal detection requires:

- **Hand position:** Hold your hand close to the camera, filling ~40-60% of the frame
- **Distance:** Arm's length from the camera
- **Background:** Plain or uncluttered backgrounds work best
- **Lighting:** Even, frontal lighting (avoid backlighting)
- **One hand only:** The model expects a single hand in the frame

### Tuning Confidence Threshold

The English confidence threshold is set to `0.45` in `config.py`. If you experience:

- **Too many false detections:** Increase `ENGLISH_CONFIDENCE_THRESHOLD` (try 0.50-0.60)
- **Too few detections:** Decrease it (try 0.35-0.40)

## Project Structure

```
├── detector.py              # YOLO + MediaPipe sign detector
├── config.py                # Configuration constants
├── app.py                   # Desktop application
├── translator.py            # Translation logic
├── sign_display.py          # Sign animation display
├── prepare_asl_dataset.py   # Convert raw ASL images to YOLO format
├── train_english.py         # Local training script
├── train_english_colab.ipynb # Colab training notebook
├── prepare_dataset.py       # Arabic dataset preparation
├── train.py                 # Arabic training script
├── train_colab.ipynb        # Arabic Colab notebook
├── models/                  # Trained model weights
├── assets/                  # Sign language images
│   ├── arabic_signs/
│   └── asl_signs/
└── static/                  # Web interface
```

## Datasets

- **Arabic:** Custom dataset with diverse subjects and backgrounds
- **English (ASL):** `grassknoted/asl-alphabet` from Kaggle (~87K images, 27 classes)

## Credits

Built with YOLOv5, MediaPipe, and OpenCV.