import os
from translator import Translator
from dotenv import load_dotenv

# Ensure env is loaded
load_dotenv()

def test_translation():
    print("Initializing Translator...")
    t = Translator()
    
    # Test simple Arabic word "مرحبا" (Hello)
    arabic_text = "م ر ح ب ا"
    print(f"Testing translation for: {arabic_text}")
    
    english = t.translate_to_english(arabic_text)
    print(f"English translation: {english}")
    
    if "hello" in english.lower():
        print("✅ Translation SUCCESS")
    else:
        print("❌ Translation might have failed or returned unexpected result")

if __name__ == "__main__":
    test_translation()
