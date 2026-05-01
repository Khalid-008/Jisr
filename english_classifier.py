"""English/ASL sign classifier using MobileNetV3-Small CNN + MediaPipe landmark MLP ensemble."""

import json
import time
import collections

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp

from config import (
    ENGLISH_CLASSIFIER_PATH,
    ENGLISH_CLASSIFIER_META_PATH,
    ENGLISH_LANDMARK_MLP_PATH,
    ENGLISH_INPUT_SIZE,
    ENGLISH_MIN_CONFIDENCE,
    ENGLISH_PREDICT_EVERY_N_FRAMES,
    ENGLISH_CONSECUTIVE_MATCHES,
    ENGLISH_CLASSES,
    WORD_PAUSE_THRESHOLD,
    SENTENCE_PAUSE_THRESHOLD,
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((ENGLISH_INPUT_SIZE, ENGLISH_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# MediaPipe landmark indices
_WRIST      = 0
_MIDDLE_MCP = 9


# ── Landmark normalisation ────────────────────────────────────────────────────

def _normalise_landmarks(lms) -> np.ndarray:
    """21-point MediaPipe landmarks → 63-float palm-relative normalised vector."""
    pts   = np.array([[l.x, l.y, l.z] for l in lms.landmark], dtype=np.float32)  # (21,3)
    pts  -= pts[_WRIST]                                                             # translate
    scale = float(np.linalg.norm(pts[_MIDDLE_MCP])) + 1e-6                        # palm length
    pts  /= scale
    return pts.flatten()                                                            # (63,)


# ── Landmark MLP definition ───────────────────────────────────────────────────

class _LandmarkMLP(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── CNN model helper ──────────────────────────────────────────────────────────

def _build_cnn(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _load_meta(meta_path: str, classes: list) -> list:
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        meta_classes = meta.get("classes", classes)
        if len(meta_classes) != len(classes):
            raise ValueError(
                f"Meta has {len(meta_classes)} classes but config expects {len(classes)}"
            )
        return meta_classes
    except FileNotFoundError:
        return classes


# ── Classifier class ──────────────────────────────────────────────────────────

class EnglishSignClassifier:
    """ASL classifier: CNN image features ensembled with a MediaPipe landmark MLP."""

    def __init__(self, model_path: str, meta_path: str, classes: list,
                 mlp_path: str | None = None):
        self.classes    = _load_meta(meta_path, classes)
        num_classes     = len(self.classes)
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN
        self._cnn = _build_cnn(num_classes)
        state = torch.load(model_path, map_location=self.device)
        self._cnn.load_state_dict(state)
        self._cnn.to(self.device).eval()

        # Landmark MLP (optional — gracefully skipped if file is absent)
        self._mlp: _LandmarkMLP | None = None
        mlp_path = mlp_path or ENGLISH_LANDMARK_MLP_PATH
        try:
            ckpt = torch.load(mlp_path, map_location=self.device)
            self._mlp = _LandmarkMLP(num_classes)
            self._mlp.load_state_dict(ckpt["state_dict"])
            self._mlp.to(self.device).eval()
            import logging
            logging.getLogger(__name__).info("Landmark MLP loaded — ensemble mode active.")
        except FileNotFoundError:
            import logging
            logging.getLogger(__name__).info(
                "Landmark MLP not found (%s) — running CNN-only mode.", mlp_path
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Could not load landmark MLP: %s", exc)

        # MediaPipe
        _mp_hands = mp.solutions.hands
        self.hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp_drawing      = mp.solutions.drawing_utils
        self._mp_hands_module = _mp_hands

        self._ring: collections.deque = collections.deque(maxlen=ENGLISH_CONSECUTIVE_MATCHES)
        self._frame_count    = 0
        self._last_committed: str | None = None

        self._live_label: str  = ""
        self._live_conf: float = 0.0

        self._last_detection_time     = 0.0
        self._word_boundary_fired     = False
        self._sentence_boundary_fired = False

        self.detected_text = ""
        self.word_count    = 0

    # ── Guide box ─────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_guide_box(frame: np.ndarray) -> tuple[np.ndarray, tuple]:
        h, w   = frame.shape[:2]
        side   = int(min(h, w) * 0.55)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - side // 2, cy - side // 2
        x2, y2 = x1 + side,      y1 + side
        return frame, (x1, y1, x2, y2)

    # ── Hand detection + crop ─────────────────────────────────────────────────

    def _detect_hand(self, frame: np.ndarray):
        """Run MediaPipe and return (crop_rgb, landmarks, annotated, hand_present)."""
        frame_h, frame_w = frame.shape[:2]
        annotated, _     = self._draw_guide_box(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = self.hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return None, None, annotated, False

        lms = result.multi_hand_landmarks[0]
        self._mp_drawing.draw_landmarks(annotated, lms, self._mp_hands_module.HAND_CONNECTIONS)

        xs = [l.x for l in lms.landmark]
        ys = [l.y for l in lms.landmark]
        x_min = max(0, int(min(xs) * frame_w))
        y_min = max(0, int(min(ys) * frame_h))
        x_max = min(frame_w, int(max(xs) * frame_w))
        y_max = min(frame_h, int(max(ys) * frame_h))

        mx = int((x_max - x_min) * 0.2)
        my = int((y_max - y_min) * 0.2)
        x_min = max(0, x_min - mx);    y_min = max(0, y_min - my)
        x_max = min(frame_w, x_max + mx); y_max = min(frame_h, y_max + my)

        # Expand to square using surrounding pixels
        bw, bh = x_max - x_min, y_max - y_min
        if bh > bw:
            d = bh - bw
            x_min = max(0, x_min - d//2);  x_max = min(frame_w, x_max + d - d//2)
        elif bw > bh:
            d = bw - bh
            y_min = max(0, y_min - d//2);  y_max = min(frame_h, y_max + d - d//2)

        crop = frame_rgb[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None, None, annotated, False

        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return crop, lms, annotated, True

    # ── CNN + MLP ensemble classification ────────────────────────────────────

    @torch.inference_mode()
    def _classify(self, crop_rgb: np.ndarray, lms) -> tuple[str, float]:
        # CNN forward pass
        tensor    = _preprocess(crop_rgb).unsqueeze(0).to(self.device)
        probs_cnn = torch.softmax(self._cnn(tensor), dim=1)[0].cpu().numpy()

        if self._mlp is None:
            idx = int(probs_cnn.argmax())
            return self.classes[idx], float(probs_cnn[idx])

        # Landmark MLP forward pass
        feat     = _normalise_landmarks(lms)
        feat_t   = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        probs_lm = torch.softmax(self._mlp(feat_t), dim=1)[0].cpu().numpy()

        # ── Asymmetric ensemble: MLP as veto, not equal voter ─────────────────
        # The two models live on very different confidence scales:
        #   • CNN is typically ~0.99 on its top class (peaky distribution).
        #   • MLP spreads mass across geometrically similar letters and sits
        #     around 0.20–0.50 even when correct (broader distribution).
        #
        # Plain averaging dilutes both confident agreement (L: 0.99+0.20→0.59)
        # AND requiring "both top picks identical" misses cases where the MLP
        # ranks the right letter #2.  So instead we treat the MLP as a
        # tie-breaker that can override the CNN only with strong evidence:
        #
        #   1. If MLP also picks CNN's top → it's a confirmation, use max conf.
        #   2. Else if MLP is highly confident in a DIFFERENT letter AND
        #      assigns very little mass to CNN's pick → trust the MLP (this is
        #      how W-misread-as-F gets corrected).
        #   3. Otherwise the MLP isn't sure → trust the CNN unchanged.
        cnn_idx, mlp_idx = int(probs_cnn.argmax()), int(probs_lm.argmax())
        cnn_conf         = float(probs_cnn[cnn_idx])
        mlp_conf         = float(probs_lm[mlp_idx])
        mlp_on_cnn_pick  = float(probs_lm[cnn_idx])

        if cnn_idx == mlp_idx:
            return self.classes[cnn_idx], max(cnn_conf, mlp_conf)

        # MLP strongly disagrees: very confident in something else AND
        # gives almost no support to the CNN's pick.  Numbers tuned so that
        # confident-but-wrong CNN picks get overridden, while ordinary
        # CNN-right / MLP-uncertain cases pass through untouched.
        if mlp_conf > 0.40 and mlp_on_cnn_pick < 0.10:
            return self.classes[mlp_idx], mlp_conf

        return self.classes[cnn_idx], cnn_conf

    # ── Main per-frame entry point ────────────────────────────────────────────

    def detect_frame(self, frame: np.ndarray):
        current_time = time.time()
        self._frame_count += 1

        crop, lms, annotated, hand_present = self._detect_hand(frame)

        detected_label       = None
        is_word_boundary     = False
        is_sentence_boundary = False

        if not hand_present or crop is None:
            self._ring.clear()
            self._last_committed = None
            self._live_label     = ""
            self._live_conf      = 0.0
            time_since = current_time - self._last_detection_time
            if self._last_detection_time > 0:
                if time_since > SENTENCE_PAUSE_THRESHOLD:
                    if not self._sentence_boundary_fired:
                        is_sentence_boundary = True
                        self._sentence_boundary_fired = True
                elif time_since > WORD_PAUSE_THRESHOLD:
                    if not self._word_boundary_fired:
                        is_word_boundary = True
                        self._word_boundary_fired = True
                        if self.detected_text and not self.detected_text.endswith(" "):
                            self.detected_text += " "
                            self.word_count = 0
            return annotated, detected_label, is_word_boundary, is_sentence_boundary

        if self._frame_count % ENGLISH_PREDICT_EVERY_N_FRAMES == 0:
            label, conf = self._classify(crop, lms)
            self._live_label = label
            self._live_conf  = conf

            if conf >= ENGLISH_MIN_CONFIDENCE and label != "NOTHING":
                if label != self._last_committed:
                    self._last_committed = None
                self._ring.append(label)
            else:
                self._ring.clear()

            if (len(self._ring) == ENGLISH_CONSECUTIVE_MATCHES
                    and len(set(self._ring)) == 1
                    and self._ring[0] != self._last_committed):
                committed = self._ring[0]
                self._ring.clear()
                self._last_committed = committed

                self._word_boundary_fired     = False
                self._sentence_boundary_fired = False

                if committed == "DEL":
                    if self.detected_text:
                        self.detected_text = self.detected_text[:-1]
                    detected_label = committed
                elif committed == "SPACE":
                    if not self.detected_text.endswith(" "):
                        self.detected_text += " "
                        self.word_count += 1
                    detected_label = committed
                else:
                    self.detected_text += committed
                    self.word_count    += 1
                    detected_label      = committed

                self._last_detection_time = current_time

        # Live confidence overlay
        if self._live_label and self._live_label != "NOTHING":
            pct   = self._live_conf * 100
            color = (0,255,0) if self._live_conf >= 0.90 else \
                    (0,200,255) if self._live_conf >= 0.70 else (0,0,255)
            cv2.putText(annotated, f"{self._live_label}  {pct:.0f}%",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return annotated, detected_label, is_word_boundary, is_sentence_boundary

    def get_detected_text(self) -> str:
        return " ".join(self.detected_text.split())

    def get_arabic_text(self) -> str:
        return self.get_detected_text()

    def clear(self):
        self._ring.clear()
        self._frame_count    = 0
        self._last_committed = None
        self._live_label     = ""
        self._live_conf      = 0.0
        self._last_detection_time     = 0.0
        self._word_boundary_fired     = False
        self._sentence_boundary_fired = False
        self.detected_text = ""
        self.word_count    = 0


def create_english_detector() -> EnglishSignClassifier:
    return EnglishSignClassifier(
        model_path=ENGLISH_CLASSIFIER_PATH,
        meta_path=ENGLISH_CLASSIFIER_META_PATH,
        classes=ENGLISH_CLASSES,
    )
