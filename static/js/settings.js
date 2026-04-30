/**
 * settings.js — Settings modal + localStorage persistence.
 */

const SettingsModule = (() => {
  const STORAGE_KEY = "signbridge_settings";
  const DEFAULTS = { apiKey: "", model: "gpt-4o", animationSpeed: 800 };

  let modal     = null;
  let apiInput  = null;
  let modelInput = null;
  let slider    = null;
  let sliderVal = null;

  function init() {
    modal      = document.getElementById("settingsModal");
    apiInput   = document.getElementById("apiKeyInput");
    modelInput = document.getElementById("modelInput");
    slider     = document.getElementById("animSpeedSlider");
    sliderVal  = document.getElementById("animSpeedValue");

    // Nav settings link
    document.getElementById("navSettingsBtn").addEventListener("click", (e) => {
      e.preventDefault();
      open();
    });

    // Close buttons
    document.getElementById("modalClose").addEventListener("click", close);
    document.getElementById("btnCancelSettings").addEventListener("click", close);

    // Overlay click (outside card) closes modal
    modal.addEventListener("click", (e) => {
      if (e.target === modal) close();
    });

    // Escape key closes modal
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modal.classList.contains("is-open")) close();
    });

    // Save button
    document.getElementById("btnSaveSettings").addEventListener("click", _save);

    // Slider live preview
    slider.addEventListener("input", () => {
      sliderVal.textContent = `${slider.value}ms`;
    });

    // Load persisted settings
    _load();
  }

  function open() {
    // Populate inputs from current saved state
    const saved = _getSaved();
    apiInput.value   = saved.apiKey;
    modelInput.value = saved.model;
    slider.value     = saved.animationSpeed;
    sliderVal.textContent = `${saved.animationSpeed}ms`;

    modal.classList.add("is-open");
    document.body.style.overflow = "hidden";
    apiInput.focus();
  }

  function close() {
    modal.classList.remove("is-open");
    document.body.style.overflow = "";
  }

  async function _save() {
    const apiKey = apiInput.value.trim();
    const model  = modelInput.value.trim() || "gpt-4o";
    const speed  = parseInt(slider.value, 10);

    if (!apiKey) {
      showToast("Please enter your OpenAI API key.", "error");
      apiInput.focus();
      return;
    }

    const saveBtn = document.getElementById("btnSaveSettings");
    saveBtn.disabled = true;
    saveBtn.textContent = "Saving…";

    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey, model }),
      });
      const data = await res.json();

      if (data.success) {
        // Persist to localStorage
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ apiKey, model, animationSpeed: speed }));
        // Propagate to app state
        EventBus.emit("settings:saved", { apiKey, model, animationSpeed: speed });
        showToast("Settings saved.", "success");
        close();
      } else {
        showToast("Failed to save settings.", "error");
      }
    } catch (err) {
      showToast(`Network error: ${err.message}`, "error");
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = "Save Settings";
    }
  }

  function _load() {
    const saved = _getSaved();
    // Emit so AppState can initialise with persisted values
    EventBus.emit("settings:loaded", saved);
  }

  function _getSaved() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : { ...DEFAULTS };
    } catch {
      return { ...DEFAULTS };
    }
  }

  return { init, open, close };
})();
