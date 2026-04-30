"""ASL Sign Display Widget for PyQt5."""

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from config import ASL_SIGNS_DIR, ASL_ANIMATION_SPEED_MS


class ASLSignDisplay(QtWidgets.QWidget):
    """Widget that displays ASL fingerspelling images letter by letter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.characters = []
        self.current_index = 0
        self.is_playing = False

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Current letter label
        self.letter_label = QtWidgets.QLabel("ASL Sign Mode")
        self.letter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.letter_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(self.letter_label)

        # Sign image display
        self.sign_image_label = QtWidgets.QLabel()
        self.sign_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sign_image_label.setMinimumSize(200, 200)
        self.sign_image_label.setStyleSheet(
            "background-color: white; border: 2px solid #ccc; border-radius: 10px;"
        )
        layout.addWidget(self.sign_image_label)

        # Progress label (e.g., "H E L L O" with current highlighted)
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 16px; color: #666;")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)

        # Animation timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._next_sign)

    def play_signs(self, characters):
        """Start displaying ASL signs for the given character list.

        Args:
            characters: List of characters like ['H', 'E', 'L', 'L', 'O', 'SPACE', 'W', ...]
        """
        self.stop()
        self.characters = characters
        self.current_index = 0

        if not characters:
            self.letter_label.setText("No signs to display")
            return

        self.is_playing = True
        self._show_current_sign()
        self.timer.start(ASL_ANIMATION_SPEED_MS)

    def stop(self):
        """Stop the animation."""
        self.timer.stop()
        self.is_playing = False

    def _next_sign(self):
        """Advance to the next sign."""
        self.current_index += 1
        if self.current_index >= len(self.characters):
            self.stop()
            self.letter_label.setText("Complete!")
            return
        self._show_current_sign()

    def _show_current_sign(self):
        """Display the current sign."""
        if self.current_index >= len(self.characters):
            return

        char = self.characters[self.current_index]

        if char == "SPACE":
            self.letter_label.setText("[SPACE]")
            self.sign_image_label.clear()
            self.sign_image_label.setText("  ")
        else:
            self.letter_label.setText(f"Letter: {char}")
            self._load_sign_image(char)

        # Update progress
        self._update_progress()

    def _load_sign_image(self, letter):
        """Load and display the ASL sign image for a letter."""
        image_path = os.path.join(ASL_SIGNS_DIR, f"{letter}.png")

        if os.path.exists(image_path):
            pixmap = QtGui.QPixmap(image_path)
            scaled = pixmap.scaled(
                180, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.sign_image_label.setPixmap(scaled)
        else:
            # Fallback: show letter text with description
            self.sign_image_label.setText(
                f"<div style='text-align:center; font-size:48px; color:#2196F3;'>"
                f"<b>{letter}</b></div>"
                f"<div style='text-align:center; font-size:12px; color:#666;'>"
                f"ASL sign for '{letter}'</div>"
            )

    def _update_progress(self):
        """Update the progress display showing all letters with current highlighted."""
        parts = []
        for i, char in enumerate(self.characters):
            display = " " if char == "SPACE" else char
            if i == self.current_index:
                parts.append(f"<b style='color:#2196F3; font-size:20px;'>{display}</b>")
            elif i < self.current_index:
                parts.append(f"<span style='color:#999;'>{display}</span>")
            else:
                parts.append(f"<span style='color:#333;'>{display}</span>")
        self.progress_label.setText(" ".join(parts))

    def replay(self):
        """Replay the current sign sequence from the beginning."""
        if self.characters:
            self.play_signs(self.characters)

    def clear(self):
        """Reset the display."""
        self.stop()
        self.characters = []
        self.current_index = 0
        self.letter_label.setText("ASL Sign Mode")
        self.sign_image_label.clear()
        self.progress_label.setText("")
