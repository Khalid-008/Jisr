"""
Generate Arabic Sign Language Reference Images
Creates placeholder images for each Arabic sign letter.
These show the Arabic letter and a brief hand-shape description.

For production use, replace these with actual Arabic Sign Language
hand-sign photographs or illustrations from a licensed source.
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join("assets", "arabic_signs")

# Arabic sign hand-shape descriptions for each letter
ARABIC_SIGN_DESCRIPTIONS = {
    "ع": "Ain — أصابع مفتوحة للأسفل",
    "ال": "Al — السبابة للأعلى مع الإبهام",
    "ا": "Alef — قبضة مع الإبهام للأعلى",
    "ب": "Ba — يد مسطحة، السبابة ممدودة",
    "د": "Dal — السبابة تشير للأعلى",
    "ظ": "Dha — قبضة مع الإبهام بارز",
    "ض": "Dhad — أصابع مضمومة، الإبهام فوق",
    "ف": "Fa — حلقة بالسبابة والإبهام",
    "ق": "Qaf — السبابة والإبهام متقابلان",
    "غ": "Ghain — يد مقوسة مفتوحة",
    "ه": "Ha — أصابع مضمومة تشير للأمام",
    "ح": "Haa — يد مفتوحة مع ثني الأصابع",
    "ج": "Jeem — قبضة مع السبابة معقوفة",
    "ك": "Kaf — يد مفتوحة مع ضم الأصابع",
    "خ": "Khaa — أصابع ممدودة متفرقة",
    "لا": "La — السبابة والوسطى متقاطعتان",
    "ل": "Lam — شكل L بالإبهام والسبابة",
    "م": "Meem — قبضة مع الإبهام تحت الأصابع",
    "ن": "Nun — قبضة مع السبابة والإبهام",
    "ر": "Ra — السبابة معقوفة",
    "ص": "Saad — قبضة مغلقة مع الإبهام",
    "س": "Seen — ثلاثة أصابع ممدودة",
    "ش": "Sheen — أربعة أصابع ممدودة متفرقة",
    "ت": "Ta — قبضة مع الإبهام بين السبابة والوسطى",
    "ط": "Taa — يد مسطحة مع ثني الإبهام",
    "ث": "Thaa — ثلاثة أصابع مع الإبهام",
    "ذ": "Thal — السبابة ممدودة مع الإبهام",
    "ة": "Taa Marbouta — حلقة بالأصابع",
    "و": "Waw — قبضة مع الخنصر ممدود",
    "ي": "Ya — الخنصر والإبهام ممدودان",
    "ى": "Alef Maqsura — يد مسطحة للأسفل",
    "ز": "Zay — السبابة ترسم Z",
}

# Colors for each letter (cycling through a palette)
COLORS = [
    "#E53935", "#D81B60", "#8E24AA", "#5E35B1",
    "#3949AB", "#1E88E5", "#039BE5", "#00ACC1",
    "#00897B", "#43A047", "#7CB342", "#C0CA33",
    "#FDD835", "#FFB300", "#FB8C00", "#F4511E",
    "#6D4C41", "#757575", "#546E7A", "#E53935",
    "#D81B60", "#8E24AA", "#5E35B1", "#3949AB",
    "#1E88E5", "#039BE5", "#757575", "#00897B",
    "#43A047", "#7CB342", "#C0CA33", "#FDD835",
]


def generate_sign_image(letter, description, color, size=200):
    """Generate a single Arabic sign placeholder image."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Try to use a font that supports Arabic
    try:
        title_font = ImageFont.truetype("arial.ttf", 64)
        desc_font = ImageFont.truetype("arial.ttf", 13)
        label_font = ImageFont.truetype("arial.ttf", 12)
    except (IOError, OSError):
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
            desc_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
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

    # Arabic letter in center (white on colored circle)
    bbox = draw.textbbox((0, 0), letter, font=title_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2 - 15
    draw.text((x, y), letter, fill="white", font=title_font)

    # Description at bottom (just the transliteration part)
    desc_short = description.split("—")[0].strip() if "—" in description else description
    bbox = draw.textbbox((0, 0), desc_short, font=desc_font)
    tw = bbox[2] - bbox[0]
    x = (size - tw) // 2
    draw.text((x, size - 30), desc_short, fill="#333333", font=desc_font)

    # "ArSL" label at top
    draw.text((5, 5), "ArSL", fill="#999999", font=label_font)

    return img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating Arabic Sign Language reference images...")
    for i, (letter, desc) in enumerate(ARABIC_SIGN_DESCRIPTIONS.items()):
        color = COLORS[i % len(COLORS)]
        img = generate_sign_image(letter, desc, color)
        path = os.path.join(OUTPUT_DIR, f"{letter}.png")
        img.save(path)
        print(f"  Created: {path}")

    # Also create a SPACE image
    space_img = generate_sign_image("فراغ", "Space — فراغ بين الكلمات", "#757575")
    space_path = os.path.join(OUTPUT_DIR, "SPACE.png")
    space_img.save(space_path)
    print(f"  Created: {space_path}")

    total = len(ARABIC_SIGN_DESCRIPTIONS) + 1
    print(f"\nGenerated {total} images in {OUTPUT_DIR}/")
    print("\nNote: These are placeholder images showing letter + description.")
    print("For production use, replace with actual Arabic sign photos.")


if __name__ == "__main__":
    main()
