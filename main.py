# ============================================================
# KAITEXY AI
# GEMINI VISION SIGN LANGUAGE BACKEND
# ============================================================
#
# Flutter
#    |
#    | POST /predict-sign
#    | multipart/form-data
#    | file = image
#    v
# FastAPI
#    |
#    v
# Gemini Vision
#    |
#    v
# Sign prediction
#    |
#    v
# JSON
# {
#   "prediction": "A",
#   "confidence": 0.94,
#   "status": "Prediction successful"
# }
#    |
#    v
# Flutter
#
# IMPORTANT:
# Gemini performs the actual image recognition.
# No PyTorch.
# No MediaPipe.
# No .pt model.
# No 63-landmark processing.
#
# ============================================================

import asyncio
import json
import os
import re
from typing import Literal, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Kaitexy AI Gemini Sign Language Backend"

APP_VERSION = "8.0-GEMINI"

# ------------------------------------------------------------
# Gemini API key
# ------------------------------------------------------------

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

# ------------------------------------------------------------
# Gemini model
# ------------------------------------------------------------

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()

# ------------------------------------------------------------
# Supported labels
# ------------------------------------------------------------

LABELS = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

# ------------------------------------------------------------
# Confidence threshold
#
# IMPORTANT:
#
# Flutter receives Gemini's confidence.
#
# We use 0.80 as the backend acceptance threshold.
#
# However, we DO NOT hide the confidence from Flutter.
# Flutter will receive the actual confidence value.
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CONFIDENCE_THRESHOLD",
        "0.80"
    )
)

# ------------------------------------------------------------
# Maximum uploaded image size
# ------------------------------------------------------------

MAX_IMAGE_BYTES = 8 * 1024 * 1024


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client: Optional[genai.Client] = None

GEMINI_READY = False


# ============================================================
# INITIALIZE GEMINI
# ============================================================

def initialize_gemini() -> bool:

    global gemini_client
    global GEMINI_READY

    print()
    print("=" * 70)
    print("INITIALIZING GEMINI")
    print("=" * 70)

    if not GEMINI_API_KEY:

        print(
            "ERROR: GEMINI_API_KEY is not set."
        )

        gemini_client = None
        GEMINI_READY = False

        return False

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        GEMINI_READY = True

        print(
            f"Gemini initialized successfully."
        )

        print(
            f"Model: {GEMINI_MODEL}"
        )

        print("=" * 70)

        return True

    except Exception as error:

        print(
            "Gemini initialization failed:"
        )

        print(
            repr(error)
        )

        gemini_client = None
        GEMINI_READY = False

        return False


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=APP_NAME,

    version=APP_VERSION,

    description=(
        "Kaitexy AI backend using "
        "Gemini Vision for sign recognition."
    )
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

    allow_headers=["*"]
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print()
    print("=" * 70)
    print("STARTING KAITEXY AI BACKEND")
    print("=" * 70)

    initialize_gemini()

    print()

    if GEMINI_READY:

        print(
            "STATUS: GEMINI READY"
        )

    else:

        print(
            "STATUS: GEMINI NOT READY"
        )

    print("=" * 70)
    print()


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():

    global gemini_client
    global GEMINI_READY

    print(
        "Shutting down Kaitexy AI..."
    )

    gemini_client = None

    GEMINI_READY = False


# ============================================================
# GEMINI STRUCTURED RESPONSE
# ============================================================

class SignResult(BaseModel):

    prediction: Literal[
        "A", "B", "C", "D", "E", "F",
        "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X",
        "Y", "Z", "UNKNOWN"
    ] = Field(
        description=(
            "Exactly one uppercase ASL "
            "fingerspelling alphabet letter "
            "from A to Z, or UNKNOWN."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence of the visual prediction "
            "from 0.0 to 1.0."
        )
    )


# ============================================================
# GEMINI SIGN RECOGNITION PROMPT
# ============================================================

SIGN_PROMPT = r"""
You are the visual sign-language recognition engine
for Kaitexy AI.

YOUR TASK:

Look at the supplied image and identify the SINGLE
STATIC ASL FINGERSPELLING ALPHABET LETTER shown
by the hand.

SUPPORTED LETTERS:

A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z

============================================================
VISUAL ANALYSIS
============================================================

Carefully inspect:

1. Number of visible fingers.
2. Which fingers are extended.
3. Which fingers are bent.
4. Thumb position.
5. Finger-to-thumb contact.
6. Finger separation.
7. Palm orientation.
8. Back-of-hand orientation.
9. Hand rotation.
10. Overall hand shape.
11. Relative position of the fingers.
12. Whether the image actually contains a hand.

============================================================
IGNORE
============================================================

Ignore:

- Face
- Hair
- Clothes
- Background
- Objects
- Written text
- Logos
- Watermarks
- Camera interface
- Decorations

Only the hand gesture matters.

============================================================
IMPORTANT
============================================================

This is STATIC ASL FINGERSPELLING.

Do NOT predict:

- a word
- a sentence
- a spoken-language word
- multiple letters
- multiple guesses

Return exactly ONE alphabet letter.

If the hand is:

- not visible
- severely cropped
- too blurry
- obstructed
- ambiguous
- not a recognizable static ASL alphabet sign

return:

UNKNOWN

============================================================
CONFIDENCE
============================================================

Confidence must represent how certain you are that the
visible hand gesture corresponds to the predicted letter.

Use:

0.90 - 1.00
Very clear and highly reliable gesture.

0.80 - 0.89
Clear gesture with good confidence.

0.60 - 0.79
Some uncertainty.

0.40 - 0.59
Weak or ambiguous evidence.

0.00 - 0.39
Very uncertain / essentially unknown.

Do NOT artificially increase confidence.

If the image is ambiguous, return UNKNOWN rather than
inventing a letter.

============================================================
FINAL OUTPUT
============================================================

Return only the structured response required by the API.
"""


# ============================================================
# MIME TYPE VALIDATION
# ============================================================

SUPPORTED_MIME_TYPES = {

    "image/jpeg": "image/jpeg",

    "image/jpg": "image/jpeg",

    "image/png": "image/png",

    "image/webp": "image/webp"

}


# ============================================================
# NORMALIZE GEMINI RESULT
# ============================================================

def normalize_prediction(
    prediction,
    confidence
):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if not isinstance(
        prediction,
        str
    ):

        return None, 0.0

    prediction = (
        prediction
        .strip()
        .upper()
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    try:

        score = float(
            confidence
        )

    except Exception:

        score = 0.0

    score = max(
        0.0,
        min(
            1.0,
            score
        )
    )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if prediction == "UNKNOWN":

        return None, score

    # --------------------------------------------------------
    # Remove accidental formatting
    # --------------------------------------------------------

    prediction = re.sub(
        r"[^A-Z]",
        "",
        prediction
    )

    # --------------------------------------------------------
    # Must be exactly one letter
    # --------------------------------------------------------

    if (
        len(prediction) != 1
        or prediction not in LABELS
    ):

        return None, score

    return prediction, score


# ============================================================
# GEMINI VISION PREDICTION
# ============================================================

async def recognize_sign(
    image_bytes: bytes,
    mime_type: str
):

    if not GEMINI_READY:

        return (
            None,
            0.0,
            "Gemini model not ready"
        )

    if gemini_client is None:

        return (
            None,
            0.0,
            "Gemini client unavailable"
        )

    try:

        # ----------------------------------------------------
        # Create image part
        # ----------------------------------------------------

        image_part = types.Part.from_bytes(

            data=image_bytes,

            mime_type=mime_type
        )

        # ----------------------------------------------------
        # Gemini request
        # ----------------------------------------------------
        #
        # Gemini receives:
        #
        #   IMAGE
        #   +
        #   SIGN PROMPT
        #
        # Gemini performs the recognition.
        #
        # ----------------------------------------------------

        response = await asyncio.to_thread(

            gemini_client.models.generate_content,

            model=GEMINI_MODEL,

            contents=[
                image_part,
                SIGN_PROMPT
            ],

            config=types.GenerateContentConfig(

                # Structured JSON output
                response_mime_type="application/json",

                response_schema=SignResult,

                # Small because response is tiny
                max_output_tokens=100,

                # Deterministic visual classification
                temperature=0.0
            )
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print()
        print("-" * 70)
        print("GEMINI PREDICTION")
        print("-" * 70)

        raw_text = getattr(
            response,
            "text",
            ""
        ) or ""

        print(
            "Gemini raw response:",
            repr(raw_text)
        )

        # ----------------------------------------------------
        # Try SDK parsed response
        # ----------------------------------------------------

        parsed = getattr(
            response,
            "parsed",
            None
        )

        if parsed is not None:

            print(
                "Structured response received."
            )

            prediction = getattr(
                parsed,
                "prediction",
                None
            )

            confidence = getattr(
                parsed,
                "confidence",
                0.0
            )

            letter, score = normalize_prediction(

                prediction,

                confidence
            )

            print(
                "Prediction:",
                prediction
            )

            print(
                "Confidence:",
                score
            )

            # ------------------------------------------------
            # UNKNOWN
            # ------------------------------------------------

            if letter is None:

                return (
                    None,
                    score,
                    "Hand not confidently recognized"
                )

            # ------------------------------------------------
            # Confidence threshold
            # ------------------------------------------------

            if score < CONFIDENCE_THRESHOLD:

                return (
                    None,
                    score,
                    "Low confidence"
                )

            return (
                letter,
                score,
                "Prediction successful"
            )

        # ----------------------------------------------------
        # FALLBACK: parse response.text
        # ----------------------------------------------------

        if raw_text:

            try:

                data = json.loads(
                    raw_text
                )

                prediction = data.get(
                    "prediction"
                )

                confidence = data.get(
                    "confidence",
                    0.0
                )

                letter, score = normalize_prediction(

                    prediction,

                    confidence
                )

                print(
                    "Fallback prediction:",
                    prediction
                )

                print(
                    "Fallback confidence:",
                    score
                )

                if letter is None:

                    return (
                        None,
                        score,
                        "Hand not confidently recognized"
                    )

                if score < CONFIDENCE_THRESHOLD:

                    return (
                        None,
                        score,
                        "Low confidence"
                    )

                return (
                    letter,
                    score,
                    "Prediction successful"
                )

            except Exception as error:

                print(
                    "Structured JSON parsing error:",
                    repr(error)
                )

        # ----------------------------------------------------
        # No usable response
        # ----------------------------------------------------

        return (
            None,
            0.0,
            "Invalid Gemini response"
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("GEMINI ERROR")
        print("=" * 70)

        print(
            repr(error)
        )

        print("=" * 70)

        return (
            None,
            0.0,
            "Gemini recognition failed"
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {

        "message":
            "Kaitexy AI Gemini Backend is running.",

        "version":
            APP_VERSION,

        "model":
            GEMINI_MODEL,

        "gemini_ready":
            GEMINI_READY,

        "prediction_endpoint":
            "/predict-sign",

        "health_endpoint":
            "/health",

        "recognition":
            "Gemini Vision",

        "supported_labels":
            LABELS
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {

        "status":
            "healthy",

        "service":
            APP_NAME,

        "version":
            APP_VERSION,

        "gemini_ready":
            GEMINI_READY,

        "gemini_model":
            GEMINI_MODEL,

        "recognition_engine":
            "Gemini Vision",

        "supported_labels":
            len(LABELS),

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "max_image_size_mb":
            MAX_IMAGE_BYTES / (
                1024 * 1024
            )
    }


# ============================================================
# PREDICT SIGN
# ============================================================
#
# THIS IS THE ENDPOINT USED BY YOUR CURRENT FLUTTER CODE.
#
# Flutter sends:
#
# POST
# /predict-sign
#
# multipart:
#
# file = image
#
# Backend returns:
#
# {
#     "prediction": "A",
#     "confidence": 0.94,
#     "status": "Prediction successful"
# }
#
# ============================================================

@app.post("/predict-sign")
async def predict_sign(
    file: UploadFile = File(...)
):

    print()
    print("=" * 70)
    print("NEW SIGN PREDICTION REQUEST")
    print("=" * 70)

    try:

        # ====================================================
        # 1. CHECK GEMINI
        # ====================================================

        if not GEMINI_READY:

            print(
                "Gemini is not ready."
            )

            return JSONResponse(

                status_code=503,

                content={

                    "prediction": "",

                    "confidence": 0.0,

                    "status":
                        "Gemini model not ready"
                }
            )

        # ====================================================
        # 2. CHECK MIME TYPE
        # ====================================================

        content_type = (
            file.content_type or ""
        ).lower()

        print(
            "Received MIME type:",
            content_type
        )

        if (
            content_type
            not in SUPPORTED_MIME_TYPES
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

        mime_type = SUPPORTED_MIME_TYPES[
            content_type
        ]

        # ====================================================
        # 3. READ IMAGE
        # ====================================================

        image_bytes = await file.read()

        print(
            "Image size:",
            len(image_bytes),
            "bytes"
        )

        # ====================================================
        # 4. EMPTY IMAGE
        # ====================================================

        if not image_bytes:

            return JSONResponse(

                status_code=400,

                content={

                    "prediction": "",

                    "confidence": 0.0,

                    "status":
                        "Empty image"
                }
            )

        # ====================================================
        # 5. IMAGE SIZE
        # ====================================================

        if (
            len(image_bytes)
            > MAX_IMAGE_BYTES
        ):

            return JSONResponse(

                status_code=413,

                content={

                    "prediction": "",

                    "confidence": 0.0,

                    "status":
                        "Image too large"
                }
            )

        # ====================================================
        # 6. SEND IMAGE TO GEMINI
        # ====================================================

        (
            letter,
            confidence,
            status
        ) = await recognize_sign(

            image_bytes,

            mime_type
        )

        # ====================================================
        # 7. GEMINI DID NOT RECOGNIZE
        # ====================================================

        if letter is None:

            print(
                "No accepted prediction."
            )

            print(
                "Confidence:",
                confidence
            )

            print(
                "Status:",
                status
            )

            return {

                "prediction": "",

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "status":
                    status
            }

        # ====================================================
        # 8. SUCCESS
        # ====================================================

        print(
            "FINAL PREDICTION:",
            letter
        )

        print(
            "FINAL CONFIDENCE:",
            confidence
        )

        print(
            "FINAL STATUS:",
            "Prediction successful"
        )

        print("=" * 70)

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

    # ========================================================
    # GENERAL SERVER ERROR
    # ========================================================

    except Exception as error:

        print()
        print("=" * 70)
        print("PREDICTION ENDPOINT ERROR")
        print("=" * 70)

        print(
            repr(error)
        )

        print("=" * 70)

        return JSONResponse(

            status_code=500,

            content={

                "prediction": "",

                "confidence": 0.0,

                "status":
                    "Server error",

                "error":
                    str(error)
            }
        )


# ============================================================
# RENDER
# ============================================================
#
# Render start command:
#
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# ============================================================