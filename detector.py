"""Sign Language Detection using YOLOv5 — Arabic path only."""

import time
import math
import cv2
from ultralytics import YOLO
from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, MAX_DETECTIONS,
    MIN_SIGN_HOLD, MOVEMENT_THRESHOLD, MAX_DETECTION_AREA,
    WORD_PAUSE_THRESHOLD, SENTENCE_PAUSE_THRESHOLD,
    ARABIC_CLASSES, ARABIC_WORD_SIGNS
)
import mediapipe as mp


class SignDetector:
    """Detects sign language gestures from webcam frames.

    Language-agnostic: works for Arabic or English depending on the
    model_path, classes, and word_signs passed to the constructor.
    """

    def __init__(self, model_path, classes, word_signs=None, confidence=None):
        """Initialize the sign detector.

        Args:
            model_path: Path to the trained YOLO model
            classes: List of class names
            word_signs: List of signs that represent whole words
            confidence: Confidence threshold (defaults to language-specific config)
        """
        self.classes = classes
        self.word_signs = word_signs or []

        self.model = YOLO(model_path)
        self.model.conf = confidence if confidence is not None else CONFIDENCE_THRESHOLD
        self.model.max_det = MAX_DETECTIONS

        # Accepted sign state
        self.last_class_id = None
        self.last_detection_time = 0

        # Pending (candidate) sign state
        self.pending_class_id = None
        self.pending_still_since = 0
        self.pending_last_cx = None
        self.pending_last_cy = None

        # One-shot boundary flags
        self._word_boundary_fired = False
        self._sentence_boundary_fired = False

        self.detected_text = ""
        self.word_count = 0

        # MediaPipe Hands for validation
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_frame(self, frame):
        """Run detection on a single frame.

        Returns:
            annotated_frame: Frame with bounding boxes drawn
            detected_label: The detected letter/word, or None
            is_word_boundary: True if a word pause was detected
            is_sentence_boundary: True if a sentence pause was detected
        """
        results = self.model(frame, verbose=False, imgsz=640)
        detections = results[0].boxes

        annotated_frame = results[0].plot()

        detected_label = None
        is_word_boundary = False
        is_sentence_boundary = False
        current_time = time.time()

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h

        valid_detection = False

        # 1. Run MediaPipe Hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)

        hx_min, hy_min, hx_max, hy_max = 0, 0, 0, 0
        is_hand_present = False

        if hand_results.multi_hand_landmarks:
            is_hand_present = True
            hand_landmarks = hand_results.multi_hand_landmarks[0]

            self.mp_drawing.draw_landmarks(
                annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            hx_min, hy_min = frame_w, frame_h
            for lm in hand_landmarks.landmark:
                hx_min = min(hx_min, int(lm.x * frame_w))
                hy_min = min(hy_min, int(lm.y * frame_h))
                hx_max = max(hx_max, int(lm.x * frame_w))
                hy_max = max(hy_max, int(lm.y * frame_h))

            margin_x = int((hx_max - hx_min) * 0.2)
            margin_y = int((hy_max - hy_min) * 0.2)
            hx_min = max(0, hx_min - margin_x)
            hy_min = max(0, hy_min - margin_y)
            hx_max = min(frame_w, hx_max + margin_x)
            hy_max = min(frame_h, hy_max + margin_y)

        if len(detections) > 0 and is_hand_present:
            box = detections[0]
            class_id = int(box.cls[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / frame_area

            cx = ((x1 + x2) / 2) / frame_w
            cy = ((y1 + y2) / 2) / frame_h
            cx_px = int(cx * frame_w)
            cy_px = int(cy * frame_h)

            if area_ratio <= MAX_DETECTION_AREA:
                if hx_min <= cx_px <= hx_max and hy_min <= cy_px <= hy_max:
                    valid_detection = True

            if valid_detection:
                if class_id < len(self.classes):
                    label = self.classes[class_id]
                else:
                    label = results[0].names[class_id]

                cx = ((x1 + x2) / 2) / frame_w
                cy = ((y1 + y2) / 2) / frame_h

                if class_id != self.pending_class_id:
                    self.pending_class_id = class_id
                    self.pending_still_since = current_time
                    self.pending_last_cx = cx
                    self.pending_last_cy = cy
                    self._word_boundary_fired = False
                    self._sentence_boundary_fired = False
                else:
                    if self.pending_last_cx is not None:
                        movement = math.hypot(cx - self.pending_last_cx,
                                              cy - self.pending_last_cy)
                        if movement > MOVEMENT_THRESHOLD:
                            self.pending_still_since = current_time

                    self.pending_last_cx = cx
                    self.pending_last_cy = cy

                    still_duration = current_time - self.pending_still_since
                    if still_duration >= MIN_SIGN_HOLD and class_id != self.last_class_id:
                        detected_label = label
                        self.last_class_id = class_id
                        self.last_detection_time = current_time

                        if label in self.word_signs:
                            if self.detected_text:
                                self.detected_text += " "
                            self.detected_text += label
                            self.word_count += 1
                        else:
                            self.detected_text += label
                            self.word_count += 1

        if not valid_detection:
            self.last_class_id = None
            self.pending_class_id = None
            self.pending_last_cx = None
            self.pending_last_cy = None

            time_since_last = current_time - self.last_detection_time
            if self.last_detection_time > 0:
                if time_since_last > SENTENCE_PAUSE_THRESHOLD:
                    if not self._sentence_boundary_fired:
                        is_sentence_boundary = True
                        self._sentence_boundary_fired = True
                elif time_since_last > WORD_PAUSE_THRESHOLD:
                    if not self._word_boundary_fired:
                        is_word_boundary = True
                        self._word_boundary_fired = True
                        if self.detected_text and not self.detected_text.endswith(" "):
                            self.detected_text += " "
                            self.word_count = 0

        return annotated_frame, detected_label, is_word_boundary, is_sentence_boundary

    def get_detected_text(self):
        """Get the accumulated detected text."""
        return " ".join(self.detected_text.split())

    # Backward compatibility alias for desktop app.py
    def get_arabic_text(self):
        return self.get_detected_text()

    def clear(self):
        """Reset all state."""
        self.last_class_id = None
        self.last_detection_time = 0
        self.pending_class_id = None
        self.pending_still_since = 0
        self.pending_last_cx = None
        self.pending_last_cy = None
        self._word_boundary_fired = False
        self._sentence_boundary_fired = False
        self.detected_text = ""
        self.word_count = 0


# Factory functions
def create_arabic_detector():
    """Create a detector for Arabic Sign Language."""
    return SignDetector(
        model_path=MODEL_PATH,
        classes=ARABIC_CLASSES,
        word_signs=ARABIC_WORD_SIGNS
    )


from english_classifier import create_english_detector  # noqa: F401


# Backward compatibility wrapper for desktop app.py
def ArabicSignDetector(model_path=None):
    """Create an Arabic sign detector (backward compatible)."""
    return SignDetector(
        model_path=model_path or MODEL_PATH,
        classes=ARABIC_CLASSES,
        word_signs=ARABIC_WORD_SIGNS
    )
