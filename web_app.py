"""
FastAPI web application for Bidirectional Sign Language Translator.
Supports Arabic → English and English → Arabic translation directions.
"""
import os
import sys
import asyncio
import base64
import json
import logging

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Ensure the project root is on sys.path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from detector import create_arabic_detector, create_english_detector
from translator import Translator
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Sign Language Translator")

# Mount static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Shared singletons (created once at startup)
# ---------------------------------------------------------------------------
arabic_detector = None
english_detector = None
active_detector = None
current_direction = "ar2en"  # "ar2en" | "en2ar"
translator_instance: Optional[Translator] = None
detector_lock = asyncio.Lock()


@app.on_event("startup")
async def startup_event():
    global arabic_detector, english_detector, active_detector, translator_instance

    # Arabic detector (always expected to be available)
    logger.info("Initializing Arabic detector...")
    try:
        arabic_detector = create_arabic_detector()
        logger.info("Arabic detector ready.")
    except Exception as e:
        logger.warning(f"Arabic detector init warning: {e}")
        arabic_detector = None

    # English detector (may not be trained yet)
    logger.info("Initializing English detector...")
    try:
        english_detector = create_english_detector()
        logger.info("English detector ready.")
    except Exception as e:
        logger.warning(f"English detector not available (model may not be trained yet): {e}")
        english_detector = None

    active_detector = arabic_detector

    logger.info("Initializing translator...")
    translator_instance = Translator()
    logger.info("Translator ready.")


@app.on_event("shutdown")
async def shutdown_event():
    for det in [arabic_detector, english_detector]:
        if det and hasattr(det, "hands") and det.hands:
            try:
                det.hands.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    mode: str = "text"  # "text" | "signs"
    direction: str = "ar2en"  # "ar2en" | "en2ar"


class TranslateResponse(BaseModel):
    translated: str
    characters: List[str]
    mode: str


class DirectionRequest(BaseModel):
    direction: str  # "ar2en" | "en2ar"


class SettingsRequest(BaseModel):
    api_key: str
    model: str = "gpt-4o"


class SettingsResponse(BaseModel):
    success: bool
    message: str


class TTSRequest(BaseModel):
    text: str
    language: str = "en"  # "en" | "ar"


class TranscribeResponse(BaseModel):
    text: str
    language: str


class ClearResponse(BaseModel):
    success: bool


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/api/status")
async def get_status():
    """Report which models are loaded and current direction."""
    return {
        "arabic_model_loaded": arabic_detector is not None,
        "english_model_loaded": english_detector is not None,
        "direction": current_direction,
    }


@app.post("/api/direction")
async def set_direction(req: DirectionRequest):
    """Switch the active detection direction."""
    global active_detector, current_direction

    if req.direction not in ("ar2en", "en2ar"):
        raise HTTPException(status_code=400, detail="Invalid direction. Use 'ar2en' or 'en2ar'.")

    async with detector_lock:
        if req.direction == "en2ar":
            if english_detector is None:
                raise HTTPException(
                    status_code=400,
                    detail="English model not trained yet. Place english_sign_classifier.pt in models/ directory."
                )
            active_detector = english_detector
        else:
            if arabic_detector is None:
                raise HTTPException(status_code=400, detail="Arabic model not available.")
            active_detector = arabic_detector

        active_detector.clear()
        current_direction = req.direction

    return {"success": True, "direction": current_direction}


@app.get("/api/signs/{letter}")
async def get_sign_image(letter: str):
    """Serve ASL fingerspelling PNG images."""
    letter = letter.upper()
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"SPACE"}
    if letter not in allowed:
        raise HTTPException(status_code=400, detail="Invalid letter")

    signs_dir = os.path.join(os.path.dirname(__file__), config.ASL_SIGNS_DIR)
    for ext, mime in [("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("png", "image/png")]:
        image_path = os.path.join(signs_dir, f"{letter}.{ext}")
        if os.path.exists(image_path):
            return FileResponse(image_path, media_type=mime, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail=f"Sign image not found for '{letter}'")


@app.get("/api/arabic-signs/{letter}")
async def get_arabic_sign_image(letter: str):
    """Serve Arabic sign language images (jpg preferred, png fallback)."""
    signs_dir = os.path.join(os.path.dirname(__file__), config.ARABIC_SIGNS_DIR)
    for ext, mime in [("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("png", "image/png")]:
        image_path = os.path.join(signs_dir, f"{letter}.{ext}")
        if os.path.exists(image_path):
            return FileResponse(image_path, media_type=mime, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail=f"Arabic sign image not found for '{letter}'")


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    """Translate detected text based on direction."""
    if not req.text.strip():
        return TranslateResponse(translated="", characters=[], mode=req.mode)

    loop = asyncio.get_event_loop()

    if req.direction == "en2ar":
        # English → Arabic
        if req.mode == "signs":
            arabic, characters = await loop.run_in_executor(
                None, translator_instance.translate_for_arabic_signs, req.text
            )
            return TranslateResponse(translated=arabic, characters=characters, mode="signs")
        else:
            arabic = await loop.run_in_executor(
                None, translator_instance.translate_to_arabic, req.text
            )
            return TranslateResponse(translated=arabic, characters=[], mode="text")
    else:
        # Arabic → English (default)
        if req.mode == "signs":
            english, characters = await loop.run_in_executor(
                None, translator_instance.translate_for_signs, req.text
            )
            return TranslateResponse(translated=english, characters=characters, mode="signs")
        else:
            english = await loop.run_in_executor(
                None, translator_instance.translate_to_english, req.text
            )
            return TranslateResponse(translated=english, characters=[], mode="text")


@app.post("/api/settings", response_model=SettingsResponse)
async def update_settings(req: SettingsRequest):
    """Update OpenAI API key and model at runtime."""
    if not req.api_key.strip():
        raise HTTPException(status_code=422, detail="API key cannot be empty")
    translator_instance.set_api_key(req.api_key.strip())
    translator_instance.set_model(req.model.strip())
    return SettingsResponse(success=True, message="Settings updated successfully")


@app.post("/api/clear", response_model=ClearResponse)
async def clear_state():
    """Reset active detector accumulated text and state."""
    async with detector_lock:
        if active_detector:
            active_detector.clear()
    return ClearResponse(success=True)


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Generate speech audio using OpenAI TTS. Supports English and Arabic."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty")

    api_key = translator_instance._api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")

    # Choose voice based on language
    voice = "nova" if req.language == "en" else "alloy"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        audio_bytes = response.content
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Transcribe a recorded audio blob using OpenAI Whisper. Auto-detects ar/en when language is omitted."""
    api_key = translator_instance._api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Audio payload is empty")
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio exceeds 25 MB limit")

    lang_hint = language.strip().lower() if language and language.strip() else None
    if lang_hint not in (None, "ar", "en"):
        lang_hint = None

    filename = audio.filename or "recording.webm"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        kwargs = {
            "model": "whisper-1",
            "file": (filename, audio_bytes, audio.content_type or "audio/webm"),
            "response_format": "verbose_json",
            # Prompt nudges Whisper away from punctuation so signs display cleanly
            "prompt": "بدون علامات ترقيم no punctuation marks",
        }
        if lang_hint:
            kwargs["language"] = lang_hint
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.audio.transcriptions.create(**kwargs)
        )
        text = (getattr(result, "text", "") or "").strip()
        detected = getattr(result, "language", None) or lang_hint or ""
        if detected.startswith("ar"):
            detected = "ar"
        elif detected.startswith("en"):
            detected = "en"
        return TranscribeResponse(text=text, language=detected)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")


# ---------------------------------------------------------------------------
# WebSocket: real-time detection stream
# ---------------------------------------------------------------------------
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "connected", "message": "Detector ready"}))
    logger.info("WebSocket client connected")

    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = msg.get("type")

            if msg_type == "clear":
                async with detector_lock:
                    if active_detector:
                        active_detector.clear()
                await websocket.send_text(json.dumps({"type": "cleared"}))

            elif msg_type == "frame":
                b64_data = msg.get("data", "")
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]

                try:
                    img_bytes = base64.b64decode(b64_data)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Failed to decode frame")
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": f"Frame decode error: {e}"}))
                    continue

                try:
                    async with detector_lock:
                        annotated, label, word_boundary, sentence_boundary = await loop.run_in_executor(
                            None, active_detector.detect_frame, frame
                        )
                    detected_text = active_detector.get_detected_text()
                except Exception as e:
                    logger.error(f"Detection error: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "message": f"Detection error: {e}"}))
                    continue

                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

                response = {
                    "type": "detection",
                    "frame": frame_b64,
                    "label": label,
                    "word_boundary": word_boundary,
                    "sentence_boundary": sentence_boundary,
                    "detected_text": detected_text,
                    "direction": current_direction,
                }
                await websocket.send_text(json.dumps(response))

            else:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Unknown message type: {msg_type}"}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
