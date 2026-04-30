/**
 * text-to-sign.js — Convert typed text into sign language animation.
 * Arabic text → Arabic signs, English text → ASL signs.
 * Reuses ASLDisplayModule for the actual animation.
 */

const TextToSignModule = (() => {
  const ARABIC_MULTI = ["لا"];
  const ARABIC_DIACRITICS = /[\u064B-\u065F]/g;

  let inputEl    = null;
  let langHintEl = null;

  function init() {
    inputEl    = document.getElementById("t2sInput");
    langHintEl = document.getElementById("t2sLangHint");

    if (!inputEl) return;

    inputEl.addEventListener("input", _updateHint);
    document.getElementById("btnT2SConvert").addEventListener("click", convert);
    document.getElementById("btnT2SClear").addEventListener("click", clearInput);
  }

  function convert() {
    const text = inputEl.value.trim();
    if (!text) return;

    const lang = _detectLanguage(text);
    if (!lang) {
      langHintEl.textContent = "لا توجد أحرف يمكن تحويلها.";
      return;
    }

    const chars = lang === "ar" ? _decomposeArabic(text) : _decomposeEnglish(text);
    if (!chars.length) {
      langHintEl.textContent = lang === "ar" ? "لم يُعثر على أحرف عربية." : "No letters found.";
      return;
    }

    const endpoint = lang === "ar" ? "/api/arabic-signs" : "/api/signs";
    ASLDisplayModule.setSignEndpoint(endpoint);
    ASLDisplayModule.play(chars, 800);
  }

  function clearInput() {
    inputEl.value = "";
    langHintEl.textContent = "";
    ASLDisplayModule.clear();
  }

  // ── Language detection ─────────────────────────────────────

  function _detectLanguage(text) {
    if (/[\u0600-\u06FF]/.test(text)) return "ar";
    if (/[a-zA-Z]/.test(text))        return "en";
    return null;
  }

  function _updateHint() {
    const text = inputEl.value.trim();
    if (!text) { langHintEl.textContent = ""; return; }

    const lang = _detectLanguage(text);
    if (!lang) { langHintEl.textContent = ""; return; }

    const chars = lang === "ar" ? _decomposeArabic(text) : _decomposeEnglish(text);
    const signable = chars.filter(c => c !== "SPACE").length;
    langHintEl.textContent = lang === "ar"
      ? `عربي · ${signable} حرف`
      : `English · ${signable} letter${signable !== 1 ? "s" : ""}`;
  }

  // ── Character decomposition ────────────────────────────────

  function _decomposeEnglish(text) {
    const chars = [];
    for (const ch of text.toUpperCase()) {
      if (ch >= "A" && ch <= "Z")       chars.push(ch);
      else if (ch === " " && chars.length && chars[chars.length - 1] !== "SPACE") {
        chars.push("SPACE");
      }
    }
    if (chars[chars.length - 1] === "SPACE") chars.pop();
    return chars;
  }

  function _decomposeArabic(text) {
    let clean = text.replace(ARABIC_DIACRITICS, "");
    clean = clean.replace(/[اإآٱ]/g, "أ");
    const chars = [];
    let i = 0;

    while (i < clean.length) {
      if (i + 1 < clean.length) {
        const two = clean.slice(i, i + 2);
        if (ARABIC_MULTI.includes(two)) { chars.push(two); i += 2; continue; }
      }
      const ch = clean[i];
      const code = ch.charCodeAt(0);
      if (code >= 0x0600 && code <= 0x06FF) {
        chars.push(ch);
      } else if (ch === " " && chars.length && chars[chars.length - 1] !== "SPACE") {
        chars.push("SPACE");
      }
      i++;
    }

    if (chars[chars.length - 1] === "SPACE") chars.pop();
    return chars;
  }

  return { init };
})();
