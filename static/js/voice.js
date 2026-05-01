/**
 * voice.js — Push-to-talk voice input using OpenAI Whisper.
 * Records mic audio in the browser, transcribes via /api/transcribe,
 * then delegates to TextToSignModule.convertText() for sign rendering.
 */

const VoiceModule = (() => {
  let recordBtn       = null;
  let statusEl        = null;
  let transcriptEl    = null;
  let langSelect      = null;
  let mediaRecorder   = null;
  let chunks          = [];
  let activeStream    = null;
  let isRecording     = false;

  function init() {
    recordBtn    = document.getElementById("btnRecord");
    statusEl     = document.getElementById("voiceStatus");
    transcriptEl = document.getElementById("voiceTranscript");
    langSelect   = document.getElementById("voiceLang");

    if (!recordBtn) return;

    // Convert / clear always work, even when recording isn't available
    document.getElementById("btnVoiceConvert").addEventListener("click", _convert);
    document.getElementById("btnVoiceClear").addEventListener("click", _clear);

    if (!navigator.mediaDevices || !window.MediaRecorder) {
      _setStatus("التسجيل غير متاح في هذا المتصفح / Recording not supported", true);
      recordBtn.disabled = true;
      return;
    }

    // Push-to-talk: mouse and touch
    recordBtn.addEventListener("mousedown",  _start);
    recordBtn.addEventListener("touchstart", (e) => { e.preventDefault(); _start(); }, { passive: false });
    recordBtn.addEventListener("mouseup",    _stop);
    recordBtn.addEventListener("mouseleave", _stop);
    recordBtn.addEventListener("touchend",   (e) => { e.preventDefault(); _stop(); }, { passive: false });
    recordBtn.addEventListener("touchcancel", _stop);
  }

  // ── Record ────────────────────────────────────────────────────

  async function _start() {
    if (isRecording) return;
    try {
      activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      _setStatus("تعذّر الوصول إلى الميكروفون / Microphone access denied", true);
      return;
    }

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : (MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "");

    try {
      mediaRecorder = mimeType
        ? new MediaRecorder(activeStream, { mimeType })
        : new MediaRecorder(activeStream);
    } catch {
      _setStatus("تعذّر بدء التسجيل / Failed to start recorder", true);
      _releaseStream();
      return;
    }

    chunks = [];
    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    });
    mediaRecorder.addEventListener("stop", _onStop);

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.add("recording");
    recordBtn.querySelector(".record-label").textContent = "جارٍ التسجيل... / Recording...";
    _setStatus("● يتم التسجيل... / Recording...");
  }

  function _stop() {
    if (!isRecording || !mediaRecorder) return;
    isRecording = false;
    recordBtn.classList.remove("recording");
    recordBtn.querySelector(".record-label").textContent = "اضغط مع الاستمرار للتحدث";
    try { mediaRecorder.stop(); } catch { /* no-op */ }
  }

  async function _onStop() {
    _releaseStream();
    if (!chunks.length) {
      _setStatus("لم يُسجَّل أي صوت / No audio captured", true);
      return;
    }

    const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
    chunks = [];
    _setStatus("جارٍ النسخ... / Transcribing...");

    const lang = langSelect ? langSelect.value : "";
    const form = new FormData();
    form.append("audio", blob, "recording.webm");
    if (lang) form.append("language", lang);

    try {
      const res = await fetch("/api/transcribe", { method: "POST", body: form });
      if (!res.ok) {
        const detail = await _safeError(res);
        _setStatus(`خطأ في النسخ: ${detail}`, true);
        return;
      }
      const data = await res.json();
      transcriptEl.value = data.text || "";

      if (!transcriptEl.value) {
        _setStatus("لم يُتعرّف على نص / No speech detected", true);
        return;
      }

      const langTag = data.language === "ar" ? "عربي" : (data.language === "en" ? "English" : data.language || "");
      _setStatus(`${langTag ? langTag + " · " : ""}تم النسخ — اضغط "عرض بالإشارات"`);
    } catch (err) {
      _setStatus(`خطأ في الشبكة: ${err.message}`, true);
    }
  }

  // ── Convert transcript → signs ────────────────────────────────

  function _convert() {
    const text = (transcriptEl.value || "").trim();
    if (!text) {
      _setStatus("لا يوجد نص للتحويل — سجّل صوتاً أولاً / Record audio first", true);
      return;
    }
    const ok = TextToSignModule.convertText(text, null);
    if (ok) {
      _setStatus("جارٍ عرض الإشارات / Showing signs...");
    } else {
      _setStatus("لم يُعثر على أحرف قابلة للتحويل / No convertible characters found", true);
    }
  }

  function _clear() {
    transcriptEl.value = "";
    _setStatus("");
    if (typeof ASLDisplayModule !== "undefined") ASLDisplayModule.clear();
  }

  // ── Helpers ───────────────────────────────────────────────────

  function _releaseStream() {
    if (activeStream) {
      activeStream.getTracks().forEach((t) => t.stop());
      activeStream = null;
    }
  }

  async function _safeError(res) {
    try { const j = await res.json(); return j.detail || res.statusText; }
    catch { return res.statusText; }
  }

  function _setStatus(text, isError = false) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.classList.toggle("error", !!isError);
  }

  function stopIfRecording() {
    if (isRecording) _stop();
  }

  return { init, stopIfRecording };
})();
