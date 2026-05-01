# ASL Overfitting Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix domain mismatch / overfitting in the ASL Alphabet detector so it generalizes from the unrealistic grassknoted/asl-alphabet Kaggle dataset to real webcam input.

**Architecture:** Five targeted code changes across the inference pipeline, dataset preparation, and training scripts. No model re-training required for immediate fixes (#1 and #2). Full benefit requires re-training with the updated notebook.

**Tech Stack:** YOLOv5 (Ultralytics), MediaPipe, OpenCV, Python

---

## Root Cause Analysis

The `grassknoted/asl-alphabet` dataset has ~87K images from one person, one room, near-identical lighting, plain background, and the hand fills almost the entire frame. Combined with a fallback 80%-of-frame bounding box when MediaPipe fails, the model learns "hand = whole frame" and collapses on real webcam input where the hand is small on a busy background.

### Root Causes (in priority order)

1. **Training/inference size mismatch** — Train at `imgsz=640`, infer at `imgsz=320` — degrades small-object detection
2. **Confidence threshold tuned for Arabic** — `CONFIDENCE_THRESHOLD=0.95` is too high for the ASL model
3. **Fallback bounding box** — `0.5 0.5 0.8 0.8` teaches "hand = whole frame"
4. **No augmentation** — The unrealistic dataset needs heavy augmentation to compensate
5. **No user guidance** — Users don't know optimal hand position for testing

---

## Task 1: Match Inference imgsz to Training

**Files:**
- Modify: `detector.py:68`

The model trains at `imgsz=640` but infers at `imgsz=320`. This alone degrades small-object detection meaningfully.

- [ ] **Step 1: Change inference image size**

In `detector.py`, line 68, change:

```python
results = self.model(frame, verbose=False, imgsz=320)
```

to:

```python
results = self.model(frame, verbose=False, imgsz=640)
```

- [ ] **Step 2: Verify**

Confirm the file contains `imgsz=640` on the inference line. No tests needed — this is a direct parameter change.

- [ ] **Step 3: Commit**

```bash
git add detector.py
git commit -m "fix: match inference imgsz to training (320->640)"
```

---

## Task 2: Add Per-Language Confidence Threshold

**Files:**
- Modify: `config.py` (add `ENGLISH_CONFIDENCE_THRESHOLD`)
- Modify: `detector.py` (accept per-instance confidence, update factory function)

The shared `CONFIDENCE_THRESHOLD = 0.95` works for Arabic but is too high for the ASL model, whose confidences calibrate differently.

- [ ] **Step 1: Add English-specific threshold to config.py**

```python
# Add after CONFIDENCE_THRESHOLD = 0.95
ENGLISH_CONFIDENCE_THRESHOLD = 0.45
```

- [ ] **Step 2: Update SignDetector to accept confidence parameter**

In `detector.py`, modify `__init__` to accept an optional `confidence` parameter:

```python
def __init__(self, model_path, classes, word_signs=None, confidence=None):
    # ...
    self.model.conf = confidence if confidence is not None else CONFIDENCE_THRESHOLD
```

- [ ] **Step 3: Update factory function**

In `detector.py`, modify `create_english_detector()`:

```python
def create_english_detector():
    return SignDetector(
        model_path=ENGLISH_MODEL_PATH,
        classes=ENGLISH_CLASSES,
        word_signs=ENGLISH_WORD_SIGNS,
        confidence=ENGLISH_CONFIDENCE_THRESHOLD
    )
```

- [ ] **Step 4: Update imports**

In `detector.py`, add `ENGLISH_CONFIDENCE_THRESHOLD` to the import from config.

- [ ] **Step 5: Verify**

The Arabic detector should still use `CONFIDENCE_THRESHOLD = 0.95` (default). The English detector should use `0.45`.

- [ ] **Step 6: Commit**

```bash
git add config.py detector.py
git commit -m "fix: add per-language confidence threshold (English: 0.45)"
```

---

## Task 3: Drop Fallback Bounding Box — Skip Poor-Quality Images

**Files:**
- Modify: `prepare_asl_dataset.py` (remove fallback, skip images instead)
- Modify: `train_english_colab.ipynb` (same change in Colab notebook)

When MediaPipe fails to detect a hand, the code falls back to `0.5 0.5 0.8 0.8` (80% of frame). Combined with the dataset's tight crops, this teaches the model "hand = whole frame."

- [ ] **Step 1: Update prepare_asl_dataset.py**

Remove `FALLBACK_BBOX = "0.5 0.5 0.8 0.8"`. In `process_image()`, when MediaPipe fails, return `(False, True)` instead of writing a fallback label. The image is then excluded from the dataset since `success=False` prevents it from being copied to train/val directories.

Update stats tracking from "fallback" to "skipped."

- [ ] **Step 2: Update train_english_colab.ipynb**

In the bbox generation cell, replace the fallback branch:

```python
# BEFORE:
else:
    label_line = f'{class_id} 0.5 0.5 0.8 0.8'
    class_fallback += 1

# AFTER:
else:
    class_skipped += 1
    continue
```

Also move the image copy/label write inside the `if result.multi_hand_landmarks:` block so skipped images are not copied.

- [ ] **Step 3: Verify**

Run `python prepare_asl_dataset.py --help` to confirm the script still parses correctly. Check the output mentions "skipped" instead of "fallback."

- [ ] **Step 4: Commit**

```bash
git add prepare_asl_dataset.py train_english_colab.ipynb
git commit -m "fix: drop fallback bbox — skip images where MediaPipe fails"
```

---

## Task 4: Add Strong Augmentation to Training

**Files:**
- Modify: `train_english_colab.ipynb` (Step 5 cell)
- Modify: `train_english.py`

The grassknoted dataset has zero diversity. Without augmentation, models hit 99% val mAP but collapse on real input.

- [ ] **Step 1: Add augmentation to train_english_colab.ipynb**

In the training cell (Step 5), add these parameters to `model.train()`:

```python
hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
degrees=15, translate=0.2, scale=0.6, shear=5,
perspective=0.0005,
flipud=0.0, fliplr=0.0,  # CRITICAL: don't flip - left/right hand matters for ASL
mosaic=1.0, mixup=0.15, copy_paste=0.1,
```

- [ ] **Step 2: Add augmentation to train_english.py**

Add the same parameters to the `model.train()` call in `train_english.py`.

- [ ] **Step 3: Update Colab notebook markdown**

Update the Step 5 header to mention augmentation and explain why it's needed.

- [ ] **Step 4: Verify**

Confirm both files contain the augmentation parameters. Verify `fliplr=0.0` and `flipud=0.0` are set (horizontal flipping would invert left/right hand, breaking ASL).

- [ ] **Step 5: Commit**

```bash
git add train_english_colab.ipynb train_english.py
git commit -m "fix: add strong augmentation to combat domain overfitting"
```

---

## Task 5: Document Test Conditions in README

**Files:**
- Modify: `README.md`

Users need to know how to position their hand for optimal detection until the dataset is replaced.

- [ ] **Step 1: Add ASL Detection section to README**

Include:
- Summary of the dataset limitation
- Table of fixes applied
- Testing conditions (hand position, distance, background, lighting)
- Confidence threshold tuning guide
- Training instructions

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add ASL overfitting fix documentation and test conditions"
```

---

## Post-Implementation: Re-Train the Model

After all code changes are applied, re-train the ASL model using the updated notebook:

1. Open `train_english_colab.ipynb` in Google Colab with GPU runtime
2. Run all cells (dataset preparation will skip already-processed classes)
3. The model will train with augmentation and produce a better-generalizing model
4. Download `english_sign_best.pt` and place it in `models/`

**Expected results:**
- Lower validation mAP (no more 99% on unrealistic data)
- Significantly better real-world webcam detection
- Fewer false positives from background objects

---

## Self-Review

**1. Spec coverage:** All five root causes from the diagnosis are addressed:
- [x] imgsz mismatch → Task 1
- [x] Confidence threshold → Task 2
- [x] Fallback bbox → Task 3
- [x] No augmentation → Task 4
- [x] Test conditions → Task 5

**2. Placeholder scan:** No TBDs, TODOs, or vague instructions found. Every step contains exact code.

**3. Type consistency:** `ENGLISH_CONFIDENCE_THRESHOLD` is defined in `config.py`, imported in `detector.py`, and used in `create_english_detector()`. Names are consistent throughout.

**4. Backward compatibility:** Arabic detection is unaffected — it uses the default `CONFIDENCE_THRESHOLD` and the existing `create_arabic_detector()` factory function unchanged.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-30-asl-overfitting-fix.md`.

**Status:** All tasks implemented. Model re-training required for full benefit.