"""
Arabic Sign Language to English Translation System
Main PyQt5 Application

Detects Arabic sign language via webcam, translates to English,
and outputs as text or ASL fingerspelling signs.
"""

import sys
import os
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal
import imutils


from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, WEBCAM_WIDTH, WEBCAM_HEIGHT,
    MODEL_PATH
)
from detector import ArabicSignDetector
from translator import Translator
from sign_display import ASLSignDisplay


class TranslationThread(QThread):
    """Background thread for Gemini API translation."""
    result_ready = pyqtSignal(str, str)  # (mode, result_text)
    signs_ready = pyqtSignal(str, list)  # (english_text, characters)

    def __init__(self, translator, arabic_text, mode):
        super().__init__()
        self.translator = translator
        self.arabic_text = arabic_text
        self.mode = mode

    def run(self):
        if self.mode == "text":
            english = self.translator.translate_to_english(self.arabic_text)
            self.result_ready.emit("text", english)
        else:
            english, characters = self.translator.translate_for_signs(self.arabic_text)
            self.signs_ready.emit(english, characters)


class WebcamThread(QThread):
    """Background thread for webcam capture and detection."""
    frame_ready = pyqtSignal(np.ndarray)  # annotated frame
    detection = pyqtSignal(str)  # detected label
    word_boundary = pyqtSignal()
    sentence_boundary = pyqtSignal()

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return

        self.running = True
        last_inference_time = 0
        inference_interval = 0.1  # run YOLO at max 10 FPS

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()
            if now - last_inference_time < inference_interval:
                continue

            last_inference_time = now
            annotated, label, word_bound, sentence_bound = self.detector.detect_frame(frame)
            self.frame_ready.emit(annotated)

            if label:
                self.detection.emit(label)
            if word_bound:
                self.word_boundary.emit()
            if sentence_bound:
                self.sentence_boundary.emit()

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arabic Sign Language → English Translation")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Core components
        self.detector = ArabicSignDetector()
        self.translator = Translator()
        self.webcam_thread = None
        self.translation_thread = None
        self.is_running = False
        self.translation_mode = "text"  # "text" or "sign"

        self._setup_ui()

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setSpacing(10)

        # ===== LEFT PANEL: Webcam + Arabic Text =====
        left_panel = QtWidgets.QVBoxLayout()

        # Title
        title = QtWidgets.QLabel("Arabic Sign Language Detection")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        left_panel.addWidget(title)

        # Webcam display
        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setFixedSize(WEBCAM_WIDTH, WEBCAM_HEIGHT)
        self.webcam_label.setStyleSheet(
            "background-color: #1a1a2e; border: 2px solid #16213e; border-radius: 8px;"
        )
        self.webcam_label.setAlignment(QtCore.Qt.AlignCenter)
        self.webcam_label.setText(
            "<span style='color: #e0e0e0; font-size: 16px;'>"
            "Press Start to begin detection</span>"
        )
        left_panel.addWidget(self.webcam_label)

        # Arabic text output
        arabic_label = QtWidgets.QLabel("Detected Arabic Text:")
        arabic_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 5px;")
        left_panel.addWidget(arabic_label)

        self.arabic_text_display = QtWidgets.QTextBrowser()
        self.arabic_text_display.setMaximumHeight(100)
        self.arabic_text_display.setStyleSheet(
            "font-size: 24px; padding: 8px; border: 1px solid #ccc; border-radius: 5px;"
            "background-color: #fafafa;"
        )
        self.arabic_text_display.setLayoutDirection(QtCore.Qt.RightToLeft)
        left_panel.addWidget(self.arabic_text_display)

        # Control buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 16px; "
            "padding: 10px 20px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.start_btn.clicked.connect(self._toggle_webcam)
        btn_layout.addWidget(self.start_btn)

        self.translate_btn = QtWidgets.QPushButton("Translate")
        self.translate_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-size: 16px; "
            "padding: 10px 20px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #1e88e5; }"
        )
        self.translate_btn.clicked.connect(self._translate)
        btn_layout.addWidget(self.translate_btn)

        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-size: 16px; "
            "padding: 10px 20px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.clear_btn.clicked.connect(self._clear_all)
        btn_layout.addWidget(self.clear_btn)

        left_panel.addLayout(btn_layout)
        main_layout.addLayout(left_panel)

        # ===== RIGHT PANEL: Translation Output =====
        right_panel = QtWidgets.QVBoxLayout()

        # Mode toggle
        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Translation Mode:")
        mode_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        mode_layout.addWidget(mode_label)

        self.text_mode_btn = QtWidgets.QPushButton("Text")
        self.text_mode_btn.setCheckable(True)
        self.text_mode_btn.setChecked(True)
        self.text_mode_btn.setStyleSheet(self._mode_btn_style(True))
        self.text_mode_btn.clicked.connect(lambda: self._set_mode("text"))
        mode_layout.addWidget(self.text_mode_btn)

        self.sign_mode_btn = QtWidgets.QPushButton("Sign (ASL)")
        self.sign_mode_btn.setCheckable(True)
        self.sign_mode_btn.setStyleSheet(self._mode_btn_style(False))
        self.sign_mode_btn.clicked.connect(lambda: self._set_mode("sign"))
        mode_layout.addWidget(self.sign_mode_btn)

        mode_layout.addStretch()
        right_panel.addLayout(mode_layout)

        # Direction indicator
        direction = QtWidgets.QLabel("Arabic Sign Language  →  English")
        direction.setAlignment(QtCore.Qt.AlignCenter)
        direction.setStyleSheet(
            "font-size: 16px; color: #666; padding: 5px; "
            "background-color: #e8f5e9; border-radius: 5px; margin: 5px 0;"
        )
        right_panel.addWidget(direction)

        # Stacked widget for text/sign output
        self.output_stack = QtWidgets.QStackedWidget()

        # --- Text mode page ---
        text_page = QtWidgets.QWidget()
        text_layout = QtWidgets.QVBoxLayout(text_page)

        eng_label = QtWidgets.QLabel("English Translation:")
        eng_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        text_layout.addWidget(eng_label)

        self.english_text_display = QtWidgets.QTextBrowser()
        self.english_text_display.setStyleSheet(
            "font-size: 22px; padding: 10px; border: 1px solid #ccc; "
            "border-radius: 5px; background-color: #fafafa;"
        )
        text_layout.addWidget(self.english_text_display)
        self.output_stack.addWidget(text_page)

        # --- Sign mode page ---
        sign_page = QtWidgets.QWidget()
        sign_layout = QtWidgets.QVBoxLayout(sign_page)

        self.sign_display = ASLSignDisplay()
        sign_layout.addWidget(self.sign_display)

        # English text below signs
        self.sign_english_label = QtWidgets.QLabel("")
        self.sign_english_label.setStyleSheet(
            "font-size: 18px; color: #333; padding: 8px; "
            "background-color: #e3f2fd; border-radius: 5px;"
        )
        self.sign_english_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sign_english_label.setWordWrap(True)
        sign_layout.addWidget(self.sign_english_label)

        # Repeat button
        self.repeat_btn = QtWidgets.QPushButton("Repeat Signs")
        self.repeat_btn.setStyleSheet(
            "QPushButton { background-color: #7B1FA2; color: white; font-size: 14px; "
            "padding: 8px 20px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #6A1B9A; }"
        )
        self.repeat_btn.clicked.connect(self.sign_display.replay)
        sign_layout.addWidget(self.repeat_btn)

        self.output_stack.addWidget(sign_page)
        right_panel.addWidget(self.output_stack)

        # OpenAI settings (API key + model)
        openai_box = QtWidgets.QGroupBox("OpenAI Settings")
        openai_box.setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        openai_layout = QtWidgets.QFormLayout(openai_box)
        openai_layout.setSpacing(6)

        self.api_key_input = QtWidgets.QLineEdit()
        self.api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter OpenAI API key…")
        self.api_key_input.setText(self.translator._api_key)
        self.api_key_input.setStyleSheet(
            "font-size: 12px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px;"
        )
        openai_layout.addRow("API Key:", self.api_key_input)

        self.model_input = QtWidgets.QLineEdit()
        self.model_input.setPlaceholderText("e.g. gpt-4o")
        self.model_input.setText(self.translator._model)
        self.model_input.setStyleSheet(
            "font-size: 12px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px;"
        )
        openai_layout.addRow("Model:", self.model_input)

        save_btn = QtWidgets.QPushButton("Apply")
        save_btn.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; font-size: 12px; "
            "padding: 4px 12px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #546E7A; }"
        )
        save_btn.clicked.connect(self._save_openai_settings)
        openai_layout.addRow("", save_btn)
        right_panel.addWidget(openai_box)

        # Status bar
        self.status_label = QtWidgets.QLabel("Ready. Press Start to begin.")
        self.status_label.setStyleSheet(
            "font-size: 12px; color: #999; padding: 5px; "
            "border-top: 1px solid #eee; margin-top: 5px;"
        )
        right_panel.addWidget(self.status_label)

        main_layout.addLayout(right_panel)

    def _mode_btn_style(self, active):
        if active:
            return (
                "QPushButton { background-color: #1976D2; color: white; font-size: 14px; "
                "padding: 8px 16px; border-radius: 5px; border: none; }"
            )
        return (
            "QPushButton { background-color: #e0e0e0; color: #333; font-size: 14px; "
            "padding: 8px 16px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #bdbdbd; }"
        )

    def _set_mode(self, mode):
        self.translation_mode = mode
        self.text_mode_btn.setChecked(mode == "text")
        self.sign_mode_btn.setChecked(mode == "sign")
        self.text_mode_btn.setStyleSheet(self._mode_btn_style(mode == "text"))
        self.sign_mode_btn.setStyleSheet(self._mode_btn_style(mode == "sign"))
        self.output_stack.setCurrentIndex(0 if mode == "text" else 1)

    def _toggle_webcam(self):
        if self.is_running:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        if not os.path.exists(MODEL_PATH):
            QtWidgets.QMessageBox.warning(
                self, "Model Not Found",
                f"Trained model not found at:\n{MODEL_PATH}\n\n"
                "Please train the model first using train_colab.ipynb\n"
                "and place the weights in the models/ folder."
            )
            return

        self.is_running = True
        self.start_btn.setText("Stop")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-size: 16px; "
            "padding: 10px 20px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.status_label.setText("Detecting Arabic sign language...")

        self.webcam_thread = WebcamThread(self.detector)
        self.webcam_thread.frame_ready.connect(self._update_frame)
        self.webcam_thread.detection.connect(self._on_detection)
        self.webcam_thread.word_boundary.connect(self._on_word_boundary)
        self.webcam_thread.sentence_boundary.connect(self._on_sentence_boundary)
        self.webcam_thread.start()

    def _stop_webcam(self):
        self.is_running = False
        self.start_btn.setText("Start")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 16px; "
            "padding: 10px 20px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #45a049; }"
        )

        if self.webcam_thread:
            self.webcam_thread.stop()
            self.webcam_thread = None

        self.webcam_label.clear()
        self.webcam_label.setText(
            "<span style='color: #e0e0e0; font-size: 16px;'>"
            "Press Start to begin detection</span>"
        )
        self.status_label.setText("Stopped. Press Start to resume.")

    def _update_frame(self, frame):
        """Display annotated webcam frame."""
        frame = imutils.resize(frame, width=WEBCAM_WIDTH)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _on_detection(self, label):
        """Handle a new detection."""
        # Qt handles RTL and Arabic letter shaping natively
        display_text = self.detector.get_arabic_text()

        self.arabic_text_display.clear()
        self.arabic_text_display.append(display_text)
        self.status_label.setText(f"Detected: {label}")

    def _on_word_boundary(self):
        """Auto-translate when a word pause is detected."""
        if self.detector.get_arabic_text().strip():
            self._set_status("Detection paused — translating word…", "#F57C00")
            self._translate()
            # Clear so sentence boundary (which fires 2s later) doesn't retranslate the same text
            self.detector.clear()
            self.arabic_text_display.clear()

    def _on_sentence_boundary(self):
        """Auto-translate and reset when a sentence pause is detected."""
        if self.detector.get_arabic_text().strip():
            self._set_status("Detection stopped — translating sentence…", "#C62828")
            self._translate()
            # Clear accumulated letters — ready for next sentence
            self.detector.clear()
            self.arabic_text_display.clear()

    def _translate(self):
        """Kick off a background translation — does not block the UI."""
        arabic = self.detector.get_arabic_text()
        if not arabic.strip():
            self.status_label.setText("No text to translate.")
            return

        # Avoid stacking multiple translation requests
        if self.translation_thread and self.translation_thread.isRunning():
            return

        self._set_status("Translating…", "#F57C00")
        if self.translation_thread:
            try:
                self.translation_thread.result_ready.disconnect()
                self.translation_thread.signs_ready.disconnect()
            except RuntimeError:
                pass
        self.translation_thread = TranslationThread(
            self.translator, arabic, self.translation_mode
        )
        self.translation_thread.result_ready.connect(self._on_translation_done)
        self.translation_thread.signs_ready.connect(self._on_signs_done)
        self.translation_thread.start()

    def _on_translation_done(self, _mode, english):
        """Receive text translation result from background thread."""
        self.english_text_display.clear()
        self.english_text_display.append(english)
        self._set_status("Translation complete.", "#2E7D32")

    def _on_signs_done(self, english, characters):
        """Receive sign translation result from background thread."""
        self.sign_english_label.setText(f"English: {english}")
        self.sign_display.play_signs(characters)
        self._set_status(f"Showing ASL signs for: {english}", "#2E7D32")

    def _set_status(self, text, color="#999"):
        """Update status label with colour coding."""
        self.status_label.setStyleSheet(
            f"font-size: 12px; color: {color}; padding: 5px; "
            "border-top: 1px solid #eee; margin-top: 5px;"
        )
        self.status_label.setText(text)

    def _save_openai_settings(self):
        """Apply new API key and model name immediately without restarting."""
        key = self.api_key_input.text().strip()
        model = self.model_input.text().strip()
        if not key:
            self._set_status("API key cannot be empty.", "#e53935")
            return
        if not model:
            self._set_status("Model name cannot be empty.", "#e53935")
            return
        self.translator.set_api_key(key)
        self.translator.set_model(model)
        self._set_status(f"Settings applied: {model}", "#1565C0")

    def _clear_all(self):
        """Clear all text and reset."""
        self.detector.clear()
        self.arabic_text_display.clear()
        self.english_text_display.clear()
        self.sign_display.clear()
        self.sign_english_label.setText("")
        self.status_label.setText("Cleared. Ready.")

    def closeEvent(self, event):
        """Clean up on window close."""
        if self.webcam_thread:
            self.webcam_thread.stop()
        if self.translation_thread and self.translation_thread.isRunning():
            self.translation_thread.wait()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark-ish palette for modern look
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(245, 245, 245))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(33, 33, 33))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
