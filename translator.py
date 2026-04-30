"""Arabic to English translation using OpenAI API."""

from config import OPENAI_API_KEY, OPENAI_MODEL


class Translator:
    """Translates Arabic text to English using OpenAI."""

    def __init__(self):
        self._client = None
        self._initialized = False
        self._api_key = OPENAI_API_KEY
        self._model = OPENAI_MODEL

    def set_api_key(self, key):
        """Update the API key and reset the client so it reinitialises on next use."""
        self._api_key = key.strip()
        self._client = None
        self._initialized = False

    def set_model(self, model):
        """Update the model name — takes effect on the next translation call."""
        self._model = model.strip()

    def _init_client(self):
        """Lazy initialization of OpenAI client."""
        if self._initialized:
            return True

        api_key = self._api_key
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set. Translation will not work.")
            return False

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._initialized = True
            return True
        except Exception as e:
            print(f"ERROR initializing OpenAI: {e}")
            return False

    def translate_to_english(self, arabic_text):
        """Translate Arabic text to English.

        Args:
            arabic_text: Arabic string to translate

        Returns:
            English translation string, or error message
        """
        if not arabic_text or not arabic_text.strip():
            return ""

        if not self._init_client():
            return "[Translation unavailable - check OPENAI_API_KEY]"

        try:
            prompt = (
                "You are provided with a sequence of Arabic letters that were fingerspelled one by one using sign language gestures.\n\n"
                "Task:\n"
                "1. Identify the most likely Arabic word formed strictly by these letters.\n"
                "2. Translate that Arabic word into English.\n\n"
                "Rules:\n"
                "- Do NOT add extra letters that were not provided (e.g., if provided 'أ ب', the word is 'أب' (Father), NOT 'باب' (Door)).\n"
                "- Treat all Alif variations (أ, إ, آ, ا) as the same.\n"
                "- Return ONLY the English translation, nothing else.\n\n"
                f"Provided letters: {arabic_text}"
            )
            
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a highly accurate translator specializing in Arabic Sign Language transliteration."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Translation error: {e}]"

    def translate_for_signs(self, arabic_text):
        """Translate Arabic text to English and decompose into letters.

        Returns:
            tuple: (english_text, list of individual characters including spaces)
        """
        english_text = self.translate_to_english(arabic_text)
        if english_text.startswith("["):
            return english_text, []

        # Decompose into characters for ASL fingerspelling
        characters = []
        for char in english_text.upper():
            if char.isalpha():
                characters.append(char)
            elif char == " ":
                characters.append("SPACE")
            # Skip punctuation and numbers for fingerspelling

        return english_text, characters

    def translate_to_arabic(self, english_text):
        """Translate English fingerspelled letters to Arabic.

        Args:
            english_text: English string (accumulated letters) to translate

        Returns:
            Arabic translation string, or error message
        """
        if not english_text or not english_text.strip():
            return ""

        if not self._init_client():
            return "[Translation unavailable - check OPENAI_API_KEY]"

        try:
            prompt = (
                "You are provided with a sequence of English letters that were fingerspelled one by one using ASL (American Sign Language) gestures.\n\n"
                "Task:\n"
                "1. Identify the most likely English word formed strictly by these letters.\n"
                "2. Translate that English word into Arabic.\n\n"
                "Rules:\n"
                "- Do NOT add extra letters that were not provided.\n"
                "- Return ONLY the Arabic translation, nothing else.\n\n"
                f"Provided letters: {english_text}"
            )

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a highly accurate translator specializing in ASL fingerspelling transliteration to Arabic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Translation error: {e}]"

    def translate_for_arabic_signs(self, english_text):
        """Translate English text to Arabic and decompose into characters for Arabic sign display.

        Returns:
            tuple: (arabic_text, list of individual Arabic characters including spaces)
        """
        arabic_text = self.translate_to_arabic(english_text)
        if arabic_text.startswith("["):
            return arabic_text, []

        characters = []
        for char in arabic_text:
            if '\u0600' <= char <= '\u06FF' or '\uFE70' <= char <= '\uFEFF':
                characters.append(char)
            elif char == " ":
                characters.append("SPACE")
            # Skip punctuation, diacritics for sign display

        return arabic_text, characters
