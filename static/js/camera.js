/**
 * camera.js — WebSocket client + webcam capture module.
 *
 * Display strategy:
 *  - Raw webcam video is drawn at 60fps via requestAnimationFrame
 *  - When an annotated frame arrives from the server, it replaces the raw feed
 *  - Between server responses, raw video fills in so the canvas never freezes
 *  - No canvas overlay drawn client-side
 */

const CameraModule = (() => {
  const CAPTURE_FPS  = 10;
  const JPEG_QUALITY = 0.7;
  const WS_RECONNECT_DELAY_MS = 2000;
  const WS_MAX_RETRIES = 3;

  let videoEl       = null;
  let canvasEl      = null;
  let canvasCtx     = null;
  let captureCanvas = null;
  let captureCtx    = null;
  let ws            = null;
  let captureInterval  = null;
  let rawDrawLoop      = null;
  let stream           = null;
  let _waitingForResponse  = false;
  let _reconnectAttempts   = 0;
  let _intentionallyStopped = false;

  // Latest annotated frame from server
  let _annotatedImg = null;

  function init() {
    videoEl   = document.getElementById("videoFeed");
    canvasEl  = document.getElementById("cameraCanvas");
    canvasCtx = canvasEl.getContext("2d");

    captureCanvas        = document.createElement("canvas");
    captureCanvas.width  = 640;
    captureCanvas.height = 480;
    captureCtx = captureCanvas.getContext("2d");
  }

  async function startCamera() {
    _intentionallyStopped = false;
    _reconnectAttempts    = 0;
    _annotatedImg         = null;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();
    } catch (err) {
      const msg = err.name === "NotAllowedError"
        ? "تم رفض إذن الكاميرا. يرجى السماح بالوصول وإعادة المحاولة."
        : `خطأ في الكاميرا: ${err.message}`;
      EventBus.emit("camera:error", msg);
      return;
    }

    _startRawDraw();
    _connectWebSocket();
  }

  function stopCamera() {
    _intentionallyStopped = true;
    _stopRawDraw();
    _stopCapture();
    _closeWebSocket();
    _releaseStream();
    _annotatedImg = null;

    canvasCtx.fillStyle = "#111";
    canvasCtx.fillRect(0, 0, canvasEl.width, canvasEl.height);
  }

  function sendClear() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "clear" }));
    }
    _annotatedImg = null;
  }

  // ── Draw loop ────────────────────────────────────────────────────────────
  // Shows annotated frame from server when available, raw video otherwise.

  function _startRawDraw() {
    _stopRawDraw();
    function loop() {
      if (_intentionallyStopped) return;
      if (videoEl.readyState >= 2) {
        if (_annotatedImg) {
          canvasCtx.drawImage(_annotatedImg, 0, 0, canvasEl.width, canvasEl.height);
        } else {
          canvasCtx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
        }
      }
      rawDrawLoop = requestAnimationFrame(loop);
    }
    rawDrawLoop = requestAnimationFrame(loop);
  }

  function _stopRawDraw() {
    if (rawDrawLoop) { cancelAnimationFrame(rawDrawLoop); rawDrawLoop = null; }
  }

  // ── WebSocket ────────────────────────────────────────────────────────────

  function _connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws/detect`);

    ws.onopen = () => {
      _reconnectAttempts = 0;
      EventBus.emit("ws:connected");
      _startCapture();
    };

    ws.onmessage = (evt) => _handleMessage(JSON.parse(evt.data));
    ws.onerror   = () => {};

    ws.onclose = () => {
      _stopCapture();
      EventBus.emit("ws:disconnected");
      if (!_intentionallyStopped && _reconnectAttempts < WS_MAX_RETRIES) {
        _reconnectAttempts++;
        EventBus.emit("ws:reconnecting", _reconnectAttempts);
        setTimeout(_connectWebSocket, WS_RECONNECT_DELAY_MS);
      } else if (!_intentionallyStopped) {
        EventBus.emit("ws:failed");
      }
    };
  }

  function _startCapture() {
    _stopCapture();
    captureInterval = setInterval(_captureFrame, 1000 / CAPTURE_FPS);
  }

  function _stopCapture() {
    if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
    _waitingForResponse = false;
  }

  function _captureFrame() {
    if (_waitingForResponse) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!videoEl || videoEl.readyState < 2) return;

    captureCtx.drawImage(videoEl, 0, 0, 640, 480);
    const b64 = captureCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
    _waitingForResponse = true;
    ws.send(JSON.stringify({ type: "frame", data: b64 }));
  }

  function _handleMessage(msg) {
    _waitingForResponse = false;

    if (msg.type === "detection") {
      // Update the displayed annotated frame
      const img = new Image();
      img.onload = () => { _annotatedImg = img; };
      img.src = "data:image/jpeg;base64," + msg.frame;
      EventBus.emit("detection:result", msg);
    } else if (msg.type === "cleared") {
      _annotatedImg = null;
      EventBus.emit("detection:cleared");
    } else if (msg.type === "error") {
      EventBus.emit("ws:error", msg.message);
    }
  }

  function _closeWebSocket() {
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
    _waitingForResponse = false;
  }

  function _releaseStream() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    videoEl.srcObject = null;
  }

  return { init, startCamera, stopCamera, sendClear };
})();
