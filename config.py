"""Configuration constants for Arabic Sign Language Translation System."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"

# YOLOv5 Model
MODEL_PATH = os.path.join("models", "arabic_sign_best.pt")
CONFIDENCE_THRESHOLD = 0.95
MAX_DETECTIONS = 1

# Detection timing
SAME_SIGN_COOLDOWN = 1.0  # seconds before accepting same sign again
MIN_SIGN_HOLD = 0.50        # seconds a sign must be held STILL before accepting
MOVEMENT_THRESHOLD = 0.01  # fraction of frame width; movement above this resets hold timer
MAX_DETECTION_AREA = 0.50  # ignore detections covering >20% of frame (face/body filter)
WORD_PAUSE_THRESHOLD = 2.0  # seconds of no detection = word boundary
SENTENCE_PAUSE_THRESHOLD = 4.0  # seconds of no detection = sentence boundary

# English/ASL YOLOv5 Model
ENGLISH_MODEL_PATH = os.path.join("models", "english_sign_best.pt")

# 27 ASL classes (must match training data order)
ENGLISH_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "SPACE"
]

# English word-level signs (ASL fingerspells letter-by-letter)
ENGLISH_WORD_SIGNS = []

# ASL Sign Display
ASL_SIGNS_DIR = os.path.join("assets", "asl_signs")
ARABIC_SIGNS_DIR = os.path.join("assets", "arabic_signs")
ASL_ANIMATION_SPEED_MS = 800  # milliseconds per letter in sign mode

# Arabic word-level signs (detected as whole words, not letter-by-letter)
ARABIC_WORD_SIGNS = ["أنا", "اسمي", "بحبك", "عمري", "مرحبا"]

# 32 Arabic Sign Language classes (must match training data order)
ARABIC_CLASSES = [
    "ع", "ال", "ا", "ب", "د", "ظ", "ض", "ف",
    "ق", "غ", "ه", "ح", "ج", "ك", "خ", "لا",
    "ل", "م", "ن", "ر", "ص", "س", "ش", "ت",
    "ط", "ث", "ذ", "ة", "و", "ي", "ى", "ز"
]

# Window dimensions
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 900
WEBCAM_WIDTH = 700
WEBCAM_HEIGHT = 530
