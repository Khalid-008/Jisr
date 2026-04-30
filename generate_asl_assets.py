"""
Generate ASL Alphabet Reference Images
Creates clean placeholder images for each ASL letter (A-Z).
These show the letter name and a brief hand-shape description.

For production use, replace these with actual ASL hand-sign photographs
or illustrations from a licensed source.
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join("assets", "asl_signs")

# ASL hand-shape descriptions for each letter
ASL_DESCRIPTIONS = {
    "A": "Fist, thumb beside index",
    "B": "Flat hand, fingers up, thumb tucked",
    "C": "Curved hand, like holding a ball",
    "D": "Index up, others touch thumb",
    "E": "Fingers curled, thumb tucked under",
    "F": "OK sign, 3 fingers up",
    "G": "Index & thumb point sideways",
    "H": "Index & middle point sideways",
    "I": "Pinky up, fist closed",
    "J": "Pinky up, trace J in air",
    "K": "Index & middle up, thumb between",
    "L": "L-shape: thumb & index out",
    "M": "3 fingers over thumb, fist",
    "N": "2 fingers over thumb, fist",
    "O": "Fingers curved to touch thumb",
    "P": "Like K but pointing down",
    "Q": "Like G but pointing down",
    "R": "Cross index & middle finger",
    "S": "Fist, thumb over fingers",
    "T": "Thumb between index & middle",
    "U": "Index & middle up together",
    "V": "Peace sign / V-shape",
    "W": "Index, middle, ring up spread",
    "X": "Index finger hooked",
    "Y": "Thumb & pinky out (hang loose)",
    "Z": "Index finger traces Z in air",
    "SPACE": "Pause between words",
}

# Colors for each letter (cycling through a palette)
COLORS = [
    "#E53935", "#D81B60", "#8E24AA", "#5E35B1",
    "#3949AB", "#1E88E5", "#039BE5", "#00ACC1",
    "#00897B", "#43A047", "#7CB342", "#C0CA33",
    "#FDD835", "#FFB300", "#FB8C00", "#F4511E",
    "#6D4C41", "#757575", "#546E7A", "#E53935",
    "#D81B60", "#8E24AA", "#5E35B1", "#3949AB",
    "#1E88E5", "#039BE5", "#757575",
]


def generate_sign_image(letter, description, color, size=200):
    """Generate a single ASL sign placeholder image."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        title_font = ImageFont.truetype("arial.ttf", 64)
        desc_font = ImageFont.truetype("arial.ttf", 14)
        label_font = ImageFont.truetype("arial.ttf", 12)
    except (IOError, OSError):
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
            desc_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (IOError, OSError):
            title_font = ImageFont.load_default()
            desc_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

    # Background circle
    margin = 20
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=color, outline=color
    )

    # Letter in center (white on colored circle)
    display_letter = letter if letter != "SPACE" else "SP"
    bbox = draw.textbbox((0, 0), display_letter, font=title_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2 - 15
    draw.text((x, y), display_letter, fill="white", font=title_font)

    # Description at bottom
    desc_lines = _wrap_text(description, 25)
    y_desc = size - 45
    for line in desc_lines[:2]:  # max 2 lines
        bbox = draw.textbbox((0, 0), line, font=desc_font)
        tw = bbox[2] - bbox[0]
        x = (size - tw) // 2
        draw.text((x, y_desc), line, fill="#333333", font=desc_font)
        y_desc += 18

    # "ASL" label at top
    draw.text((5, 5), "ASL", fill="#999999", font=label_font)

    return img


def _wrap_text(text, max_chars):
    """Simple text wrapping."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating ASL alphabet reference images...")
    for i, (letter, desc) in enumerate(ASL_DESCRIPTIONS.items()):
        color = COLORS[i % len(COLORS)]
        img = generate_sign_image(letter, desc, color)
        path = os.path.join(OUTPUT_DIR, f"{letter}.png")
        img.save(path)
        print(f"  Created: {path}")

    print(f"\nGenerated {len(ASL_DESCRIPTIONS)} images in {OUTPUT_DIR}/")
    print("\nNote: These are placeholder images showing letter + hand description.")
    print("For production use, replace with actual ASL hand-sign photographs.")


if __name__ == "__main__":
    main()
