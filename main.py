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
# FLAN-T5-small (lazy loaded)
#
# Optimized for:
#   - Render
#   - CPU-only deployment
#   - Low RAM environments
#   - Low persistent storage
#   - Fast startup
#
# ============================================================

import asyncio
import gc
import os
from contextlib import asynccontextmanager
from typing import Optional

# ------------------------------------------------------------
# Hugging Face cache
# ------------------------------------------------------------
# Keep downloaded HF files in a temporary/runtime directory.
# This avoids filling the application filesystem unnecessarily.
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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
# PERFORMANCE / MEMORY SETTINGS
# ============================================================

# Render CPU instances usually have limited memory.
# Restricting PyTorch threads prevents excessive RAM usage.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Kaitexy AI Sign Language Backend"
APP_VERSION = "6.1"


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.getenv(
    "SIGN_MODEL_PATH",
    "model/sign_model.pt"
)


# ============================================================
# SIGN MODEL CONFIGURATION
# ============================================================

INPUT_SIZE = 63

# IMPORTANT:
# These labels MUST match the exact class ordering used
# when sign_model.pt was trained.
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

NUM_CLASSES = len(LABELS)

CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CONFIDENCE_THRESHOLD",
        "0.60"
    )
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120


# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

MAX_NUM_HANDS = 1
MODEL_COMPLEXITY = 0

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# FLAN-T5 CONFIGURATION
# ============================================================

FLAN_MODEL_NAME = os.getenv(
    "FLAN_MODEL_NAME",
    "google/flan-t5-small"
)

FLAN_MAX_NEW_TOKENS = 40
FLAN_NUM_BEAMS = 1

# If True, FLAN-T5 is removed from RAM after every correction.
# This saves RAM but makes the next correction slower.
UNLOAD_FLAN_AFTER_REQUEST = os.getenv(
    "UNLOAD_FLAN_AFTER_REQUEST",
    "true"
).lower() == "true"


# ============================================================
# DEVICE
# ============================================================

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

# Prevent multiple requests from loading FLAN-T5 simultaneously.
flan_lock = asyncio.Lock()


# ============================================================
# PYTORCH SIGN MODEL
# ============================================================

class SignModel(nn.Module):
    """
    Neural network architecture used during training.

    IMPORTANT:
    This architecture MUST exactly match the architecture
    used to create sign_model.pt.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int
    ):
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

    global sign_model
    global SIGN_MODEL_READY

    try:

        if not os.path.isfile(MODEL_PATH):

            print(
                f"Sign model not found: {MODEL_PATH}"
            )

            SIGN_MODEL_READY = False
            return False

        print(
            f"Loading sign model: {MODEL_PATH}"
        )

        model = SignModel(
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES
        )

        # ----------------------------------------------------
        # Load weights directly into CPU memory.
        # ----------------------------------------------------

        try:

            state_dict = torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True
            )

        except TypeError:

            # Compatibility with older PyTorch.
            state_dict = torch.load(
                MODEL_PATH,
                map_location="cpu"
            )

        model.load_state_dict(state_dict)

        del state_dict
        gc.collect()

        model.to(DEVICE)
        model.eval()

        sign_model = model
        SIGN_MODEL_READY = True

        print(
            "Sign model loaded successfully "
            f"({NUM_CLASSES} classes)."
        )

        return True

    except Exception as error:

        print(
            f"Sign model loading error: {error}"
        )

        sign_model = None
        SIGN_MODEL_READY = False

        gc.collect()

        return False


# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

def initialize_mediapipe() -> bool:

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

        print(
            "MediaPipe Hands initialized."
        )

        return True

    except Exception as error:

        print(
            f"MediaPipe initialization error: {error}"
        )

        hands = None

        return False


# ============================================================
# LOAD FLAN-T5 LAZILY
# ============================================================

def load_flan_model() -> bool:

    global flan_tokenizer
    global flan_model
    global FLAN_READY

    try:

        # ----------------------------------------------------
        # Don't load it twice.
        # ----------------------------------------------------

        if (
            FLAN_READY
            and flan_model is not None
            and flan_tokenizer is not None
        ):
            return True

        print(
            f"Loading language model: "
            f"{FLAN_MODEL_NAME}"
        )

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        tokenizer = AutoTokenizer.from_pretrained(
            FLAN_MODEL_NAME,
            use_fast=True
        )

        # ----------------------------------------------------
        # Model
        #
        # low_cpu_mem_usage=True prevents unnecessary
        # temporary copies during model loading.
        # ----------------------------------------------------

        model = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_NAME,
            low_cpu_mem_usage=True
        )

        model.to(DEVICE)
        model.eval()

        flan_tokenizer = tokenizer
        flan_model = model

        FLAN_READY = True

        print(
            "FLAN-T5 initialized successfully."
        )

        return True

    except Exception as error:

        print(
            f"FLAN-T5 loading failed: {error}"
        )

        flan_tokenizer = None
        flan_model = None
        FLAN_READY = False

        gc.collect()

        return False


# ============================================================
# UNLOAD FLAN-T5
# ============================================================

def unload_flan_model():

    global flan_tokenizer
    global flan_model
    global FLAN_READY

    try:

        print(
            "Unloading FLAN-T5 from memory..."
        )

        flan_model = None
        flan_tokenizer = None

        FLAN_READY = False

        gc.collect()

        print(
            "FLAN-T5 unloaded."
        )

    except Exception as error:

        print(
            f"FLAN unload error: {error}"
        )


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting Kaitexy AI Backend")
    print("=" * 60)

    # --------------------------------------------------------
    # CORE COMPONENTS ONLY
    # --------------------------------------------------------
    #
    # FLAN-T5 is intentionally NOT loaded here.
    #
    # This dramatically reduces startup RAM usage.
    # --------------------------------------------------------

    load_sign_model()

    initialize_mediapipe()

    print(
        "FLAN-T5 will be loaded only when "
        "/correct-text is requested."
    )

    print("=" * 60)
    print("Kaitexy AI Backend startup complete")
    print("=" * 60)

    yield

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    global hands
    global sign_model

    print(
        "Shutting down Kaitexy AI..."
    )

    if hands is not None:

        try:
            hands.close()
        except Exception:
            pass

    hands = None
    sign_model = None

    unload_flan_model()

    gc.collect()

    print(
        "Shutdown complete."
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "AI-powered sign language recognition "
        "backend for Kaitexy AI."
    ),
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],

    allow_headers=["*"],
)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(
    image_bytes: bytes
):

    if hands is None:
        return None

    try:

        # ----------------------------------------------------
        # Bytes → NumPy
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
            (
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            ),
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

        results = hands.process(
            image_rgb
        )

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )

        features = []

        for landmark in hand_landmarks.landmark:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        if len(features) != INPUT_SIZE:
            return None

        return np.asarray(
            features,
            dtype=np.float32
        )

    except Exception as error:

        print(
            f"Landmark extraction error: {error}"
        )

        return None


# ============================================================
# SIGN PREDICTION
# ============================================================

def predict_landmarks(
    landmarks: np.ndarray
):

    if (
        not SIGN_MODEL_READY
        or sign_model is None
    ):
        return None, 0.0

    try:

        if landmarks is None:
            return None, 0.0

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if landmarks.shape != (
            INPUT_SIZE,
        ):
            return None, 0.0

        tensor = torch.from_numpy(
            landmarks
        ).unsqueeze(0)

        with torch.inference_mode():

            logits = sign_model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

        index = int(
            prediction.item()
        )

        score = float(
            confidence.item()
        )

        if (
            index < 0
            or index >= len(LABELS)
        ):
            return None, 0.0

        if (
            score
            < CONFIDENCE_THRESHOLD
        ):
            return None, score

        return (
            LABELS[index],
            score
        )

    except Exception as error:

        print(
            f"PyTorch prediction error: "
            f"{error}"
        )

        return None, 0.0


# ============================================================
# FLAN TEXT CORRECTION
# ============================================================

def correct_text_flan(
    text: str
) -> str:

    if not text:
        return ""

    text = text.strip()

    if not text:
        return ""

    if not FLAN_READY:
        return text

    try:

        prompt = (
            "Correct the spelling and grammar "
            "of the sentence. "
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

        # ----------------------------------------------------
        # CPU inference
        # ----------------------------------------------------

        with torch.inference_mode():

            output_tokens = (
                flan_model.generate(
                    **inputs,
                    max_new_tokens=FLAN_MAX_NEW_TOKENS,
                    num_beams=FLAN_NUM_BEAMS,
                    do_sample=False,
                    early_stopping=True,
                )
            )

        corrected = (
            flan_tokenizer.decode(
                output_tokens[0],
                skip_special_tokens=True
            )
            .strip()
        )

        del inputs
        del output_tokens

        gc.collect()

        if not corrected:
            return text

        return corrected

    except Exception as error:

        print(
            f"FLAN correction error: "
            f"{error}"
        )

        gc.collect()

        return text


# ============================================================
# REQUEST MODELS
# ============================================================

class CorrectTextRequest(
    BaseModel
):

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Text produced by the "
            "sign recognition system."
        )
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {

        "status": "healthy",

        "service": APP_NAME,

        "version": APP_VERSION,

        "sign_model_ready":
            SIGN_MODEL_READY,

        "flan_loaded":
            FLAN_READY,

        "labels":
            len(LABELS),

        "input_size":
            INPUT_SIZE,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "device":
            str(DEVICE),

        "flan_model":
            FLAN_MODEL_NAME,

        "flan_lazy_loading":
            True,

    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {

        "message":
            "Kaitexy AI Backend is running.",

        "version":
            APP_VERSION,

        "health":
            "/health",

        "prediction_endpoint":
            "/predict-sign",

        "text_endpoint":
            "/correct-text",

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
        # Model check
        # ----------------------------------------------------

        if not SIGN_MODEL_READY:

            return JSONResponse(
                status_code=503,
                content={
                    "prediction": "",
                    "confidence": 0.0,
                    "status":
                        "Sign model not ready"
                }
            )

        # ----------------------------------------------------
        # File type validation
        # ----------------------------------------------------

        content_type = (
            file.content_type or ""
        )

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/jpg",
        }

        if (
            content_type
            not in allowed_types
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "prediction": "",
                    "confidence": 0.0,
                    "status":
                        "Unsupported image type"
                }
            )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = (
            await file.read()
        )

        if not image_bytes:

            return {
                "prediction": "",
                "confidence": 0.0,
                "status":
                    "Empty image"
            }

        # ----------------------------------------------------
        # Landmark extraction
        # ----------------------------------------------------

        landmarks = (
            await asyncio.to_thread(
                extract_landmarks,
                image_bytes
            )
        )

        # Release image bytes as soon as possible.
        del image_bytes

        if landmarks is None:

            return {
                "prediction": "",
                "confidence": 0.0,
                "status":
                    "No hand detected"
            }

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        letter, confidence = (
            await asyncio.to_thread(
                predict_landmarks,
                landmarks
            )
        )

        del landmarks

        # ----------------------------------------------------
        # Low confidence
        # ----------------------------------------------------

        if letter is None:

            return {
                "prediction": "",
                "confidence":
                    round(
                        confidence,
                        4
                    ),
                "status":
                    "Low confidence"
            }

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        return {

            "prediction":
                letter,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "status":
                "Prediction successful"

        }

    except Exception as error:

        print(
            f"Predict endpoint error: "
            f"{error}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "prediction": "",
                "confidence": 0.0,
                "status":
                    "Server error",
                "error":
                    str(error),
            }
        )


# ============================================================
# TEXT CORRECTION ENDPOINT
# ============================================================

@app.post("/correct-text")
async def correct_text(
    request: CorrectTextRequest
):

    raw_text = (
        request.text.strip()
    )

    if not raw_text:

        return {
            "raw_text": "",
            "corrected_text": "",
            "status":
                "Empty text"
        }

    # --------------------------------------------------------
    # Only one FLAN loading/inference operation at a time.
    # This prevents multiple simultaneous requests from
    # creating multiple large memory allocations.
    # --------------------------------------------------------

    async with flan_lock:

        try:

            # ------------------------------------------------
            # Lazy load FLAN-T5.
            #
            # It is NOT loaded during application startup.
            # ------------------------------------------------

            if not FLAN_READY:

                loaded = (
                    await asyncio.to_thread(
                        load_flan_model
                    )
                )

                if not loaded:

                    return {
                        "raw_text":
                            raw_text,

                        "corrected_text":
                            raw_text,

                        "status":
                            "FLAN unavailable; "
                            "original text returned"
                    }

            # ------------------------------------------------
            # Correct text
            # ------------------------------------------------

            corrected = (
                await asyncio.to_thread(
                    correct_text_flan,
                    raw_text
                )
            )

            result = {

                "raw_text":
                    raw_text,

                "corrected_text":
                    corrected,

                "status":
                    "Text corrected successfully"

            }

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Unload FLAN after the request.
            #
            # This keeps Render RAM low.
            # ------------------------------------------------

            if UNLOAD_FLAN_AFTER_REQUEST:

                await asyncio.to_thread(
                    unload_flan_model
                )

            return result

        except Exception as error:

            print(
                f"Text correction error: "
                f"{error}"
            )

            # Try to release memory.
            if UNLOAD_FLAN_AFTER_REQUEST:

                await asyncio.to_thread(
                    unload_flan_model
                )

            return {

                "raw_text":
                    raw_text,

                "corrected_text":
                    raw_text,

                "status":
                    "Correction failed",

                "error":
                    str(error)

            }


# ============================================================
# END
# ============================================================
#
# Render start command:
#
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# Local:
#
# uvicorn main:app --host 0.0.0.0 --port 8000
#
# ============================================================