# ============================================================
# KAITEXY AI - SIGN LANGUAGE BACKEND
# ============================================================
#
# Architecture:
#
# Flutter
#    ↓
# FastAPI
#    ↓
# MediaPipe Hand Landmarks
#    ↓
# 63 Features
#    ↓
# PyTorch Sign Classifier
#    ↓
# Prediction + Confidence
#    ↓
# Flutter builds text
#    ↓
# /correct-text
#    ↓
# FLAN-T5-small
#
# Designed for:
#   - CPU deployment
#   - Render
#   - FastAPI
#   - PyTorch
#   - MediaPipe
#   - FLAN-T5-small
#
# ============================================================

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Kaitexy AI Sign Language Backend"
APP_VERSION = "6.0"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = os.getenv(
    "SIGN_MODEL_PATH",
    "model/sign_model.pt"
)

# ------------------------------------------------------------
# Sign model
# ------------------------------------------------------------

INPUT_SIZE = 63

# IMPORTANT:
# These labels MUST match the exact class ordering used
# during training of sign_model.pt.
#
# If your model was trained with a different ordering,
# replace this list with the training order.
#
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

NUM_CLASSES = len(LABELS)

# Minimum confidence required before accepting a prediction.
#
# This should ideally be tuned using your validation dataset.
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.60")
)

# ------------------------------------------------------------
# Image processing
# ------------------------------------------------------------

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120

# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------

MAX_NUM_HANDS = 1
MODEL_COMPLEXITY = 0

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ------------------------------------------------------------
# FLAN-T5
# ------------------------------------------------------------

FLAN_MODEL_NAME = os.getenv(
    "FLAN_MODEL_NAME",
    "google/flan-t5-small"
)

FLAN_MAX_NEW_TOKENS = 40
FLAN_NUM_BEAMS = 2

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device("cpu")


# ============================================================
# GLOBAL STATE
# ============================================================

sign_model: Optional[nn.Module] = None
flan_tokenizer = None
flan_model = None
hands = None

SIGN_MODEL_READY = False
FLAN_READY = False


# ============================================================
# PYTORCH SIGN MODEL
# ============================================================

class SignModel(nn.Module):
    """
    Neural network architecture used by the trained
    sign_model.pt.

    IMPORTANT:
    This architecture must EXACTLY match the architecture
    used during training.
    """

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),

            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# LOAD SIGN MODEL
# ============================================================

def load_sign_model() -> bool:
    """
    Load the trained PyTorch sign classifier safely.
    """

    global sign_model
    global SIGN_MODEL_READY

    try:
        if not os.path.isfile(MODEL_PATH):
            print(f"Sign model not found: {MODEL_PATH}")
            SIGN_MODEL_READY = False
            return False

        print(f"Loading sign model: {MODEL_PATH}")

        model = SignModel(
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES
        )

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()

        sign_model = model
        SIGN_MODEL_READY = True

        print(
            f" Sign model loaded successfully "
            f"({NUM_CLASSES} classes)."
        )

        return True

    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        try:
            model = SignModel(
                input_size=INPUT_SIZE,
                num_classes=NUM_CLASSES
            )

            state_dict = torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )

            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()

            sign_model = model
            SIGN_MODEL_READY = True

            print(
                f" Sign model loaded successfully "
                f"({NUM_CLASSES} classes)."
            )

            return True

        except Exception as error:
            print(f" Sign model loading error: {error}")
            SIGN_MODEL_READY = False
            return False

    except Exception as error:
        print(f" Sign model loading error: {error}")
        SIGN_MODEL_READY = False
        return False


# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

def initialize_mediapipe() -> bool:
    """
    Initialize MediaPipe Hands.
    """

    global hands

    try:
        mp_hands = mp.solutions.hands

        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=MAX_NUM_HANDS,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        print(" MediaPipe Hands initialized.")
        return True

    except Exception as error:
        print(f" MediaPipe initialization error: {error}")
        hands = None
        return False


# ============================================================
# LOAD FLAN-T5
# ============================================================

def load_flan_model() -> bool:
    """
    Load FLAN-T5-small locally.

    No external API is required for inference after
    the model has been downloaded.
    """

    global flan_tokenizer
    global flan_model
    global FLAN_READY

    try:
        print(f"Loading language model: {FLAN_MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(
            FLAN_MODEL_NAME
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_NAME
        )

        model.to(DEVICE)
        model.eval()

        flan_tokenizer = tokenizer
        flan_model = model

        FLAN_READY = True

        print(" FLAN-T5 initialized successfully.")

        return True

    except Exception as error:
        print(f" FLAN-T5 loading failed: {error}")

        flan_tokenizer = None
        flan_model = None
        FLAN_READY = False

        return False


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting Kaitexy AI Backend")
    print("=" * 60)

    # Load core AI components.
    load_sign_model()
    initialize_mediapipe()

    # FLAN is optional.
    load_flan_model()

    print("=" * 60)
    print(" Kaitexy AI Backend startup complete")
    print("=" * 60)

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    global hands
    global sign_model
    global flan_model
    global flan_tokenizer

    print("Shutting down Kaitexy AI...")

    if hands is not None:
        try:
            hands.close()
        except Exception:
            pass

    hands = None
    sign_model = None
    flan_model = None
    flan_tokenizer = None

    print("Shutdown complete.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "AI-powered sign language recognition backend "
        "for Kaitexy AI."
    ),
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    # For development/mobile applications.
    # Restrict this later if browser-only production
    # access is required.
    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(image_bytes: bytes):
    """
    Decode an image and extract 21 MediaPipe hand landmarks.

    Returns:
        np.ndarray with shape (63,)
        or None if no valid hand is detected.
    """

    if hands is None:
        return None

    try:

        # ----------------------------------------------------
        # Convert bytes → NumPy image
        # ----------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return None

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        # ----------------------------------------------------
        # First detected hand
        # ----------------------------------------------------

        hand_landmarks = results.multi_hand_landmarks[0]

        features = []

        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # ----------------------------------------------------
        # Validate feature count
        # ----------------------------------------------------

        if len(features) != INPUT_SIZE:
            return None

        return np.asarray(
            features,
            dtype=np.float32
        )

    except Exception as error:
        print(f"Landmark extraction error: {error}")
        return None


# ============================================================
# SIGN PREDICTION
# ============================================================

def predict_landmarks(landmarks: np.ndarray):
    """
    Run PyTorch inference.

    Returns:
        (label, confidence)

    If confidence is below the configured threshold,
    returns:
        (None, confidence)
    """

    if not SIGN_MODEL_READY or sign_model is None:
        return None, 0.0

    try:

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if landmarks is None:
            return None, 0.0

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if landmarks.shape != (INPUT_SIZE,):
            return None, 0.0

        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            landmarks
        ).unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.inference_mode():

            logits = sign_model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        index = int(prediction.item())
        score = float(confidence.item())

        # ----------------------------------------------------
        # Validate class index
        # ----------------------------------------------------

        if index < 0 or index >= len(LABELS):
            return None, 0.0

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        if score < CONFIDENCE_THRESHOLD:
            return None, score

        return LABELS[index], score

    except Exception as error:
        print(f"PyTorch prediction error: {error}")
        return None, 0.0


# ============================================================
# FLAN TEXT CORRECTION
# ============================================================

def correct_text_flan(text: str) -> str:
    """
    Correct spelling and grammar while preserving meaning.

    FLAN-T5 is used locally.
    """

    if not text:
        return ""

    text = text.strip()

    if not text:
        return ""

    if not FLAN_READY:
        # Graceful fallback.
        return text

    try:

        prompt = (
            "Correct the spelling and grammar of the sentence. "
            "Preserve the original meaning. "
            "Do not add new information. "
            "Do not remove important information. "
            "Return only the corrected sentence.\n\n"
            f"Sentence: {text}"
        )

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        inputs = flan_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        with torch.inference_mode():

            output_tokens = flan_model.generate(
                **inputs,
                max_new_tokens=FLAN_MAX_NEW_TOKENS,
                num_beams=FLAN_NUM_BEAMS,
                do_sample=False,
                early_stopping=True,
            )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        corrected = flan_tokenizer.decode(
            output_tokens[0],
            skip_special_tokens=True
        ).strip()

        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        if not corrected:
            return text

        return corrected

    except Exception as error:

        print(
            f"FLAN correction error: {error}"
        )

        return text


# ============================================================
# REQUEST MODELS
# ============================================================

class CorrectTextRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text produced by the sign recognition system."
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,

        "sign_model_ready": SIGN_MODEL_READY,
        "flan_enabled": FLAN_READY,

        "labels": len(LABELS),
        "input_size": INPUT_SIZE,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "device": str(DEVICE),
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "Kaitexy AI Backend is running.",
        "version": APP_VERSION,
        "health": "/health",
        "prediction_endpoint": "/predict-sign",
        "text_endpoint": "/correct-text",
    }


# ============================================================
# SIGN PREDICTION ENDPOINT
# ============================================================

@app.post("/predict-sign")
async def predict_sign(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not SIGN_MODEL_READY:

            return JSONResponse(
                status_code=503,
                content={
                    "prediction": "",
                    "confidence": 0.0,
                    "status": "Sign model not ready"
                }
            )

        # ----------------------------------------------------
        # Validate file type
        # ----------------------------------------------------

        content_type = file.content_type or ""

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/jpg",
        }

        if content_type not in allowed_types:

            return JSONResponse(
                status_code=400,
                content={
                    "prediction": "",
                    "confidence": 0.0,
                    "status": "Unsupported image type"
                }
            )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:

            return {
                "prediction": "",
                "confidence": 0.0,
                "status": "Empty image"
            }

        # ----------------------------------------------------
        # Extract landmarks
        # ----------------------------------------------------

        landmarks = await asyncio.to_thread(
            extract_landmarks,
            image_bytes
        )

        if landmarks is None:

            return {
                "prediction": "",
                "confidence": 0.0,
                "status": "No hand detected"
            }

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        letter, confidence = await asyncio.to_thread(
            predict_landmarks,
            landmarks
        )

        # ----------------------------------------------------
        # Low-confidence result
        # ----------------------------------------------------

        if letter is None:

            return {
                "prediction": "",
                "confidence": round(
                    confidence,
                    4
                ),
                "status": "Low confidence"
            }

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        return {
            "prediction": letter,
            "confidence": round(
                confidence,
                4
            ),
            "status": "Prediction successful"
        }

    except Exception as error:

        print(
            f"Predict endpoint error: {error}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "prediction": "",
                "confidence": 0.0,
                "status": "Server error",
                "error": str(error),
            }
        )


# ============================================================
# TEXT CORRECTION ENDPOINT
# ============================================================

@app.post("/correct-text")
async def correct_text(
    request: CorrectTextRequest
):

    raw_text = request.text.strip()

    if not raw_text:

        return {
            "raw_text": "",
            "corrected_text": "",
            "status": "Empty text"
        }

    try:

        corrected = await asyncio.to_thread(
            correct_text_flan,
            raw_text
        )

        return {
            "raw_text": raw_text,
            "corrected_text": corrected,
            "status": (
                "Text corrected successfully"
                if FLAN_READY
                else "FLAN unavailable; original text returned"
            )
        }

    except Exception as error:

        print(
            f"Text correction error: {error}"
        )

        return {
            "raw_text": raw_text,
            "corrected_text": raw_text,
            "status": "Correction failed",
            "error": str(error)
        }


# ============================================================
# RUN LOCALLY
# ============================================================
#
# Use:
#
#   uvicorn main:app --host 0.0.0.0 --port 8000
#
# For Render:
#
#   uvicorn main:app --host 0.0.0.0 --port $PORT
#
# ============================================================
