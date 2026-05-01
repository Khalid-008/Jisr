/**
 * app.js — Main application controller.
 * Manages AppState, EventBus, UIModule, auto-translate, direction switching.
 * Supports bidirectional translation: Arabic↔English.
 */

// ── EventBus ──────────────────────────────────────────────────────────────
const EventBus = (() => {
  const handlers = {};
  return {
    on(event, cb) {
      (handlers[event] = handlers[event] || []).push(cb);
    },
    off(event, cb) {
      if (handlers[event]) {
        handlers[event] = handlers[event].filter((h) => h !== cb);
      }
    },
    emit(event, data) {
      (handlers[event] || []).forEach((h) => h(data));
    },
  };
})();

// ── AppState ──────────────────────────────────────────────────────────────
const AppState = {
  isDetecting:       false,
  detectedText:      "",        // generic: Arabic or English depending on direction
  translatedText:    "",        // generic: English or Arabic depending on direction
  signCharacters:    [],
  translationMode:   "text",    // "text" | "signs"
  animationSpeed:    800,
  status:            "idle",    // "idle" | "detecting" | "translating" | "error"
  previousDetected:  "",        // used to detect new letters for animation
  lastTranslatedText: "",       // prevent redundant translations
  direction:         "ar2en",   // "ar2en" | "en2ar"
  englishModelLoaded: false,
};

function setState(patch) {
  Object.assign(AppState, patch);
  UIModule.render(AppState);
}

// ── Toast helper ──────────────────────────────────────────────────────────
function showToast(message, type = "success") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("fade-out");
    toast.addEventListener("animationend", () => toast.remove());
  }, 2800);
}

// ── Text-to-Speech ───────────────────────────────────────────────────────
let _ttsAudioUrl = null;

async function playTTS(text, language) {
  if (!text || text.startsWith("[")) return;

  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language: language || "en" }),
    });
    if (!res.ok) throw new Error(`TTS error ${res.status}`);

    const blob = await res.blob();
    if (_ttsAudioUrl) URL.revokeObjectURL(_ttsAudioUrl);
    _ttsAudioUrl = URL.createObjectURL(blob);

    const audio = new Audio(_ttsAudioUrl);
    audio.play();

    const btn = document.getElementById("btnPlayAgain");
    if (btn) btn.disabled = false;
  } catch (err) {
    console.error("TTS playback failed:", err);
  }
}

function replayTTS() {
  if (!_ttsAudioUrl) return;
  const audio = new Audio(_ttsAudioUrl);
  audio.play();
}

// ── Direction Switching ──────────────────────────────────────────────────
async function switchDirection(dir) {
  if (AppState.direction === dir) return;

  // Clear current state first
  await clearAllFull();

  // Tell backend to switch detector
  try {
    const res = await fetch("/api/direction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ direction: dir }),
    });

    if (!res.ok) {
      const data = await res.json();
      showToast(data.detail || "Failed to switch direction", "error");
      return;
    }

    setState({ direction: dir });
    _updateDirectionUI(dir);
  } catch (err) {
    showToast(`فشل تبديل الاتجاه: ${err.message}`, "error");
  }
}

function _updateDirectionUI(dir) {
  const isAr2En = dir === "ar2en";

  // Toggle button active state
  document.getElementById("btnAr2En").classList.toggle("active", isAr2En);
  document.getElementById("btnEn2Ar").classList.toggle("active", !isAr2En);

  // Swap labels
  document.getElementById("labelDetected").textContent =
    isAr2En ? "الإشارات المكتشفة" : "Detected Signs";
  document.getElementById("labelTranslated").textContent =
    isAr2En ? "الترجمة الإنجليزية" : "الترجمة العربية";
  document.getElementById("labelSignPanel").textContent =
    isAr2En ? "تهجئة إشارة أمريكية" : "إشارات عربية";
  document.getElementById("labelSignPlaceholder").textContent =
    isAr2En ? "ستظهر إشارات ASL هنا" : "ستظهر الإشارات العربية هنا";

  // Swap text direction on display divs
  const detectedDiv = document.getElementById("arabicText");
  const translatedDiv = document.getElementById("englishText");
  detectedDiv.dir = isAr2En ? "rtl" : "ltr";
  translatedDiv.dir = isAr2En ? "ltr" : "rtl";

  // Update mode toggle label
  document.getElementById("modeSigns").textContent =
    isAr2En ? "إشارة ASL" : "إشارة عربية";

  // Switch sign image endpoint
  ASLDisplayModule.setSignEndpoint(isAr2En ? "/api/signs" : "/api/arabic-signs");
}

// ── Translation ───────────────────────────────────────────────────────────
async function triggerTranslation(autoTriggered = false) {
  if (!AppState.detectedText.trim()) return;
  if (AppState.status === "translating") return;

  setState({ status: "translating" });
  if (!autoTriggered) showToast("جارٍ الترجمة…", "info");

  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: AppState.detectedText,
        mode: AppState.translationMode,
        direction: AppState.direction,
      }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    setState({
      translatedText:     data.translated,
      signCharacters:     data.characters || [],
      lastTranslatedText: AppState.detectedText,
      status:             "detecting",
    });

    if (AppState.translationMode === "signs" && data.characters && data.characters.length > 0) {
      ASLDisplayModule.play(data.characters, AppState.animationSpeed);
    }

    // Auto-play TTS for the translated text
    if (data.translated) {
      const lang = AppState.direction === "ar2en" ? "en" : "ar";
      playTTS(data.translated, lang);
    }

    if (!autoTriggered) showToast("اكتملت الترجمة.", "success");
  } catch (err) {
    setState({ status: "error" });
    showToast(`فشلت الترجمة: ${err.message}`, "error");
  }
}

async function clearAll() {
  await fetch("/api/clear", { method: "POST" });
  CameraModule.sendClear();

  setState({
    detectedText:      "",
    previousDetected:  "",
    lastTranslatedText: "",
    status: AppState.isDetecting ? "detecting" : "idle",
  });
}

async function clearAllFull() {
  await fetch("/api/clear", { method: "POST" });
  CameraModule.sendClear();

  setState({
    detectedText:      "",
    translatedText:    "",
    signCharacters:    [],
    previousDetected:  "",
    lastTranslatedText: "",
    status: AppState.isDetecting ? "detecting" : "idle",
  });

  ASLDisplayModule.clear();

  if (_ttsAudioUrl) { URL.revokeObjectURL(_ttsAudioUrl); _ttsAudioUrl = null; }
  const btnPlay = document.getElementById("btnPlayAgain");
  if (btnPlay) btnPlay.disabled = true;
}

// ── UIModule ──────────────────────────────────────────────────────────────
const UIModule = (() => {
  let btnStart    = null;
  let btnClear    = null;
  let btnTranslate = null;
  let detectedDiv = null;
  let translatedDiv = null;
  let statusDot   = null;
  let statusLabel = null;
  let placeholder = null;
  let modeTextBtn = null;
  let modeSignBtn = null;

  function init() {
    btnStart     = document.getElementById("btnStart");
    btnClear     = document.getElementById("btnClear");
    btnTranslate = document.getElementById("btnTranslate");
    detectedDiv  = document.getElementById("arabicText");
    translatedDiv = document.getElementById("englishText");
    statusDot    = document.getElementById("statusDot");
    statusLabel  = document.getElementById("statusLabel");
    placeholder  = document.getElementById("cameraPlaceholder");
    modeTextBtn  = document.getElementById("modeText");
    modeSignBtn  = document.getElementById("modeSigns");

    // Start / Stop
    btnStart.addEventListener("click", async () => {
      if (AppState.isDetecting) {
        CameraModule.stopCamera();
        setState({ isDetecting: false, status: "idle" });
      } else {
        setState({ isDetecting: true, status: "detecting" });
        await CameraModule.startCamera();
      }
    });

    // Clear
    btnClear.addEventListener("click", clearAllFull);

    // Translate
    btnTranslate.addEventListener("click", () => triggerTranslation(false));

    // Play Again (TTS replay)
    document.getElementById("btnPlayAgain").addEventListener("click", replayTTS);

    // Mode toggle
    modeTextBtn.addEventListener("click", () => {
      setState({ translationMode: "text" });
      _syncModeButtons();
    });
    modeSignBtn.addEventListener("click", () => {
      setState({ translationMode: "signs" });
      _syncModeButtons();
    });

    // Direction toggle
    document.getElementById("btnAr2En").addEventListener("click", () => switchDirection("ar2en"));
    document.getElementById("btnEn2Ar").addEventListener("click", () => switchDirection("en2ar"));

    // Input mode toggle (camera vs text vs voice)
    const btnInCam   = document.getElementById("inputModeCamera");
    const btnInTxt   = document.getElementById("inputModeText");
    const btnInVoice = document.getElementById("inputModeVoice");
    const cameraView = document.getElementById("cameraView");
    const textView   = document.getElementById("textView");
    const voiceView  = document.getElementById("voiceView");
    const camControls   = document.getElementById("cameraControls");
    const txtControls   = document.getElementById("textControls");
    const voiceControls = document.getElementById("voiceControls");

    function selectInputMode(mode) {
      if (mode !== "camera" && AppState.isDetecting) {
        CameraModule.stopCamera();
        setState({ isDetecting: false, status: "idle" });
      }
      if (mode !== "voice" && typeof VoiceModule !== "undefined") {
        VoiceModule.stopIfRecording();
      }

      btnInCam.classList.toggle("active", mode === "camera");
      btnInTxt.classList.toggle("active", mode === "text");
      if (btnInVoice) btnInVoice.classList.toggle("active", mode === "voice");

      cameraView.style.display = mode === "camera" ? "block" : "none";
      textView.style.display   = mode === "text"   ? "flex"  : "none";
      if (voiceView) voiceView.style.display = mode === "voice" ? "block" : "none";

      camControls.style.display   = mode === "camera" ? "flex" : "none";
      txtControls.style.display   = mode === "text"   ? "flex" : "none";
      if (voiceControls) voiceControls.style.display = mode === "voice" ? "flex" : "none";
    }

    btnInCam.addEventListener("click", () => selectInputMode("camera"));
    btnInTxt.addEventListener("click", () => selectInputMode("text"));
    if (btnInVoice) btnInVoice.addEventListener("click", () => selectInputMode("voice"));

    // Navbar scroll effect
    window.addEventListener("scroll", () => {
      document.getElementById("navbar").classList.toggle("scrolled", window.scrollY > 50);
    });
  }

  function render(state) {
    // Start/Stop button
    btnStart.textContent = state.isDetecting ? "إيقاف الكشف" : "بدء الكشف";
    btnStart.className   = state.isDetecting ? "btn-secondary" : "btn-primary";

    // Clear & Translate buttons
    btnClear.disabled    = !state.isDetecting && !state.detectedText;
    btnTranslate.disabled = !state.detectedText.trim() || state.status === "translating";
    btnTranslate.textContent = state.status === "translating" ? "جارٍ الترجمة…" : "ترجمة";

    // Camera placeholder
    if (placeholder) {
      placeholder.style.display = state.isDetecting ? "none" : "flex";
    }

    // Status dot & label
    if (state.status === "detecting" || state.isDetecting) {
      statusDot.className  = "status-dot active";
      statusLabel.textContent = state.status === "translating" ? "جارٍ الترجمة…" : "يعمل";
    } else if (state.status === "error") {
      statusDot.className  = "status-dot error";
      statusLabel.textContent = "خطأ";
    } else {
      statusDot.className  = "status-dot";
      statusLabel.textContent = "في الانتظار";
    }

    // Detected text (top box)
    _renderDetectedText(state.detectedText, state.previousDetected);

    // Translated text (bottom box)
    if (state.translatedText) {
      translatedDiv.innerHTML = "";
      translatedDiv.textContent = state.translatedText;
    } else {
      translatedDiv.innerHTML = '<span class="placeholder-text">ستظهر الترجمة هنا...</span>';
    }
  }

  function _renderDetectedText(text, previous) {
    if (!text) {
      const placeholderMsg = AppState.direction === "ar2en"
        ? "ستظهر الإشارات هنا..."
        : "Detected signs will appear here...";
      detectedDiv.innerHTML = `<span class="placeholder-text">${placeholderMsg}</span>`;
      return;
    }

    const isNewLetter = text.length > previous.length;

    if (isNewLetter) {
      const base    = text.slice(0, -1);
      const newChar = text.slice(-1);
      detectedDiv.innerHTML =
        (base ? `<span>${base}</span>` : "") +
        `<span class="arabic-letter-new">${newChar}</span>`;

      const animated = detectedDiv.querySelector(".arabic-letter-new");
      if (animated) {
        animated.addEventListener("animationend", () => {
          animated.className = "";
        }, { once: true });
      }
    } else {
      detectedDiv.innerHTML = "";
      detectedDiv.textContent = text;
    }
  }

  function _syncModeButtons() {
    modeTextBtn.classList.toggle("active", AppState.translationMode === "text");
    modeSignBtn.classList.toggle("active", AppState.translationMode === "signs");
  }

  return { init, render };
})();

// ── EventBus Wiring ───────────────────────────────────────────────────────

EventBus.on("ws:connected", () => {});

EventBus.on("ws:disconnected", () => {
  if (!AppState.isDetecting) return;
});

EventBus.on("ws:reconnecting", (attempt) => {
  showToast(`انقطع الاتصال. جارٍ إعادة الاتصال… (${attempt}/3)`, "info");
});

EventBus.on("ws:failed", () => {
  setState({ isDetecting: false, status: "error" });
  showToast("تعذّر الاتصال بخادم الكشف.", "error");
});

EventBus.on("ws:error", (msg) => {
  showToast(`خطأ في الكشف: ${msg}`, "error");
});

EventBus.on("camera:error", (msg) => {
  setState({ isDetecting: false, status: "error" });
  showToast(msg, "error");
});

EventBus.on("detection:result", (msg) => {
  const prev = AppState.detectedText;

  setState({
    previousDetected: prev,
    detectedText:     msg.detected_text || "",
    status:           "detecting",
  });

  // Auto-translate on word or sentence boundary
  const currentText = (msg.detected_text || "").trim();
  const alreadyTranslated = currentText === (AppState.lastTranslatedText || "").trim();

  if ((msg.word_boundary || msg.sentence_boundary) && currentText && !alreadyTranslated) {
    triggerTranslation(true).then(() => {
      setTimeout(clearAll, 800);
    });
  }
});

EventBus.on("detection:cleared", () => {});

// ── Bootstrap ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  CameraModule.init();
  ASLDisplayModule.init();
  TextToSignModule.init();
  if (typeof VoiceModule !== "undefined") VoiceModule.init();
  UIModule.init();

  // Check which models are available
  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const status = await res.json();
      AppState.englishModelLoaded = status.english_model_loaded;

      // Disable English direction button if model not available
      const btnEn2Ar = document.getElementById("btnEn2Ar");
      if (!status.english_model_loaded) {
        btnEn2Ar.disabled = true;
        btnEn2Ar.title = "English model not trained yet";
      }
    }
  } catch (e) {
    // Server may not be ready yet, ignore
  }

  // Initial render
  UIModule.render(AppState);
});
