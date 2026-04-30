/**
 * asl-display.js — ASL fingerspelling animation module.
 * Fetches sign images from /api/signs/{letter} and animates them
 * letter by letter with a smooth crossfade and progress chip row.
 */

const ASLDisplayModule = (() => {
  let imgEl = null;
  let letterEl = null;
  let progressEl = null;
  let placeholderEl = null;
  let letterDisplayEl = null;
  let completeEl = null;
  let replayBtn = null;

  let characters = [];
  let currentIndex = 0;
  let timerId = null;
  let speed = 1400; // ms per letter
  let _signEndpoint = "/api/signs"; // default: ASL signs

  function init() {
    imgEl          = document.getElementById("aslSignImg");
    letterEl       = document.getElementById("aslCurrentLetter");
    progressEl     = document.getElementById("aslProgress");
    placeholderEl  = document.getElementById("aslPlaceholder");
    letterDisplayEl = document.getElementById("aslLetterDisplay");
    completeEl     = document.getElementById("aslComplete");
    replayBtn      = document.getElementById("btnReplay");

    replayBtn.addEventListener("click", replay);
  }

  function play(chars, animSpeed) {
    if (!chars || chars.length === 0) return;
    stop();

    characters   = chars;
    currentIndex = 0;
    speed        = animSpeed || speed;

    _show();
    _showCurrent();
    timerId = setInterval(_advance, speed);
  }

  function stop() {
    if (timerId) {
      clearInterval(timerId);
      timerId = null;
    }
  }

  function replay() {
    if (characters.length > 0) {
      play(characters, speed);
    }
  }

  function clear() {
    stop();
    characters   = [];
    currentIndex = 0;
    _hide();
  }

  function setSpeed(ms) {
    speed = ms;
  }

  // ── Private helpers ─────────────────────────────────────────────────────

  function _show() {
    placeholderEl.style.display  = "none";
    imgEl.style.display          = "block";
    letterDisplayEl.style.display = "block";
    completeEl.style.display     = "none";
    replayBtn.disabled           = false;
  }

  function _hide() {
    placeholderEl.style.display  = "flex";
    imgEl.style.display          = "none";
    letterDisplayEl.style.display = "none";
    completeEl.style.display     = "none";
    progressEl.innerHTML         = "";
    replayBtn.disabled           = true;
  }

  function _showCurrent() {
    const char = characters[currentIndex];
    if (!char) return;

    // Update letter label
    letterEl.textContent = char === "SPACE" ? "·" : char;

    // Crossfade: fade out → swap src → fade in
    imgEl.classList.add("fade-out");
    const newSrc = `${_signEndpoint}/${encodeURIComponent(char)}`;

    const tempImg = new Image();
    tempImg.onload = () => {
      imgEl.src = newSrc;
      imgEl.alt = char;
      imgEl.classList.remove("fade-out");
    };
    tempImg.onerror = () => {
      // No image for this character — just show letter, hide image
      imgEl.src = "";
      imgEl.classList.remove("fade-out");
    };
    tempImg.src = newSrc;

    _updateProgress();
  }

  function _advance() {
    currentIndex++;
    if (currentIndex >= characters.length) {
      stop();
      _showComplete();
      return;
    }
    _showCurrent();
  }

  function _updateProgress() {
    progressEl.innerHTML = "";
    characters.forEach((char, i) => {
      const chip = document.createElement("span");
      chip.className = "asl-chip";
      chip.textContent = char === "SPACE" ? "·" : char;

      if (i < currentIndex)       chip.classList.add("done");
      else if (i === currentIndex) chip.classList.add("current");

      progressEl.appendChild(chip);
    });
  }

  function _showComplete() {
    completeEl.style.display = "block";
    _updateProgress(); // mark all as done
    // Mark all chips done
    progressEl.querySelectorAll(".asl-chip").forEach((c) => {
      c.className = "asl-chip done";
    });
  }

  function setSignEndpoint(endpoint) {
    _signEndpoint = endpoint;
  }

  return { init, play, stop, replay, clear, setSpeed, setSignEndpoint };
})();
