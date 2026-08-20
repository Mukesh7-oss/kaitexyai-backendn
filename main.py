# ============================================================
# KAITEXY AI
# GEMINI VISION WORD-LEVEL SIGN LANGUAGE BACKEND
# ============================================================
#
# Flutter
#     |
#     | POST /predict-sign
#     | multipart/form-data
#     | file = image
#     v
# FastAPI
#     |
#     v
# Gemini Vision
#     |
#     v
# Word-level sign recognition
#     |
#     v
# JSON
# {
#     "prediction": "hello",
#     "confidence": 0.94,
#     "status": "Prediction successful"
# }
#     |
#     v
# Flutter
#
# NO PyTorch
# NO MediaPipe
# NO .pt MODEL
# NO 63 LANDMARKS
# NO A-Z RESTRICTION
#
# ============================================================

import asyncio
import json
import os
import re
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Kaitexy AI Gemini Word-Level Sign Language Backend"

APP_VERSION = "9.0-GEMINI-WORD"

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()

CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CONFIDENCE_THRESHOLD",
        "0.70"
    )
)

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

        print("ERROR: GEMINI_API_KEY is not set.")

        gemini_client = None
        GEMINI_READY = False

        return False

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        GEMINI_READY = True

        print("Gemini initialized successfully.")
        print(f"Model: {GEMINI_MODEL}")

        print("=" * 70)

        return True

    except Exception as error:

        print("Gemini initialization failed:")
        print(repr(error))

        gemini_client = None
        GEMINI_READY = False

        return False


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(

    title=APP_NAME,

    version=APP_VERSION,

    description=(
        "Kaitexy AI word-level sign-language "
        "recognition using Gemini Vision."
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

        print("STATUS: GEMINI READY")

    else:

        print("STATUS: GEMINI NOT READY")

    print("=" * 70)
    print()


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():

    global gemini_client
    global GEMINI_READY

    print("Shutting down Kaitexy AI...")

    gemini_client = None
    GEMINI_READY = False


# ============================================================
# STRUCTURED GEMINI RESPONSE
# ============================================================

class SignResult(BaseModel):

    prediction: str = Field(
        description=(
            "The single most likely natural-language "
            "meaning of the visible sign-language gesture. "
            "Use a concise English word or short phrase. "
            "Return UNKNOWN when the gesture cannot be "
            "reliably interpreted."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence from 0.0 to 1.0 representing "
            "confidence that the predicted meaning is "
            "correct."
        )
    )


# ============================================================
# WORD-LEVEL SIGN RECOGNITION PROMPT
# ============================================================

SIGN_PROMPT = r"""

You are Kaitexy AI's professional visual
sign-language interpretation engine.

Your task is to analyze the supplied image and
identify the meaning of the sign-language gesture
shown by the person.

IMPORTANT:

This is WORD-LEVEL sign recognition.

Do NOT restrict yourself to alphabet letters.

Do NOT assume the gesture is A-Z fingerspelling.

The image may represent:

- a single sign
- a common sign-language word
- a common sign-language phrase
- a greeting
- an action
- a response
- a request
- an emotion
- a person/object concept
- another recognizable lexical sign

Your job is to determine the most likely
English meaning of the visible sign.

============================================================
VISUAL ANALYSIS
============================================================

Carefully analyze:

1. Hand shape.
2. Number of visible hands.
3. Finger positions.
4. Finger movement if visible.
5. Thumb position.
6. Palm orientation.
7. Back-of-hand orientation.
8. Wrist orientation.
9. Hand location relative to the body.
10. Relationship between both hands.
11. Facial expression when relevant.
12. Body posture when relevant.
13. Spatial relationship between hands and body.
14. Direction of the gesture.
15. Overall configuration of the sign.

============================================================
IMPORTANT SIGN-LANGUAGE REASONING
============================================================

Do not identify a sign using only one finger.

Consider the complete visual gesture.

Distinguish between:

- alphabet fingerspelling
- lexical signs
- numbers
- gestures
- ordinary hand movements
- non-sign gestures

If the gesture clearly represents a word-level sign,
return its most likely English meaning.

============================================================
LANGUAGE
============================================================

Return the meaning in English.

Prefer a concise form.

Examples of acceptable outputs:

hello
good
help
please
sorry
thank you
yes
no
stop
come
go
eat
drink
water
home
school
friend
love
family

These examples are NOT the complete vocabulary.

You may recognize other valid sign-language meanings.

============================================================
DO NOT INVENT
============================================================

Do not invent a sign meaning merely because
a hand is visible.

If:

- the hand is not visible
- the image is severely cropped
- the image is extremely blurry
- the gesture is obstructed
- the gesture is ambiguous
- the gesture is ordinary non-sign movement
- there is insufficient visual evidence

return:

UNKNOWN

============================================================
SINGLE PREDICTION
============================================================

Return ONE best interpretation.

Do not return:

"hello or hi"

Do not return:

"help / please"

Do not return multiple guesses.

Choose the single most likely meaning.

If no reliable interpretation exists:

UNKNOWN

============================================================
CONFIDENCE
============================================================

Confidence must reflect actual visual certainty.

0.90 - 1.00
Very clear and strongly recognizable sign.

0.80 - 0.89
Clear sign with good visual evidence.

0.70 - 0.79
Reasonably recognizable but some uncertainty.

0.50 - 0.69
Weak or ambiguous evidence.

0.00 - 0.49
Very uncertain.

Do NOT artificially increase confidence.

============================================================
FINAL REQUIREMENT
============================================================

Return only the structured response requested by
the API.

"""


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

SUPPORTED_MIME_TYPES = {

    "image/jpeg": "image/jpeg",

    "image/jpg": "image/jpeg",

    "image/png": "image/png",

    "image/webp": "image/webp",

    "image/heic": "image/heic",

    "image/heif": "image/heif"
}


# ============================================================
# NORMALIZE PREDICTION
# ============================================================

def normalize_prediction(
    prediction,
    confidence
):

    if not isinstance(
        prediction,
        str
    ):

        return None, 0.0

    prediction = prediction.strip()

    try:

        score = float(confidence)

    except Exception:

        score = 0.0

    score = max(
        0.0,
        min(
            1.0,
            score
        )
    )

    if not prediction:

        return None, score

    normalized = prediction.lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    )

    if normalized in {
        "unknown",
        "uncertain",
        "cannot determine",
        "unable to determine",
        "not recognizable",
        "unrecognizable",
        "none"
    }:

        return None, score

    return normalized, score


# ============================================================
# GEMINI VISION
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
        # IMAGE
        # ----------------------------------------------------

        image_part = types.Part.from_bytes(

            data=image_bytes,

            mime_type=mime_type
        )

        # ----------------------------------------------------
        # GEMINI REQUEST
        # ----------------------------------------------------

        response = await asyncio.to_thread(

            gemini_client.models.generate_content,

            model=GEMINI_MODEL,

            contents=[
                SIGN_PROMPT,
                image_part
            ],

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema=SignResult,

                max_output_tokens=100,

                temperature=0.0
            )
        )

        # ----------------------------------------------------
        # RAW RESPONSE
        # ----------------------------------------------------

        print()
        print("-" * 70)
        print("GEMINI WORD-LEVEL PREDICTION")
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
        # STRUCTURED RESPONSE
        # ----------------------------------------------------

        parsed = getattr(
            response,
            "parsed",
            None
        )

        if parsed is not None:

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

            word, score = normalize_prediction(
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

            if word is None:

                return (
                    None,
                    score,
                    "Sign not confidently recognized"
                )

            # ------------------------------------------------
            # THRESHOLD
            # ------------------------------------------------

            if score < CONFIDENCE_THRESHOLD:

                return (
                    None,
                    score,
                    "Low confidence"
                )

            return (
                word,
                score,
                "Prediction successful"
            )

        # ----------------------------------------------------
        # FALLBACK JSON
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

                word, score = normalize_prediction(
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

                if word is None:

                    return (
                        None,
                        score,
                        "Sign not confidently recognized"
                    )

                if score < CONFIDENCE_THRESHOLD:

                    return (
                        None,
                        score,
                        "Low confidence"
                    )

                return (
                    word,
                    score,
                    "Prediction successful"
                )

            except Exception as error:

                print(
                    "JSON parsing error:",
                    repr(error)
                )

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
            "Kaitexy AI Gemini Word-Level Backend is running.",

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
            "Gemini Vision Word-Level Sign Recognition",

        "alphabet_restriction":
            False,

        "fixed_vocabulary":
            False
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

        "recognition_level":
            "word",

        "alphabet_restriction":
            False,

        "fixed_vocabulary":
            False,

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

@app.post("/predict-sign")
async def predict_sign(
    file: UploadFile = File(...)
):

    print()
    print("=" * 70)
    print("NEW WORD-LEVEL SIGN REQUEST")
    print("=" * 70)

    try:

        # ====================================================
        # 1. GEMINI
        # ====================================================

        if not GEMINI_READY:

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
        # 2. MIME TYPE
        # ====================================================

        content_type = (
            file.content_type or ""
        ).lower()

        print(
            "Received MIME type:",
            content_type
        )

        if content_type not in SUPPORTED_MIME_TYPES:

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
        # 3. IMAGE
        # ====================================================

        image_bytes = await file.read()

        print(
            "Image size:",
            len(image_bytes),
            "bytes"
        )

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
        # 4. SIZE
        # ====================================================

        if len(image_bytes) > MAX_IMAGE_BYTES:

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
        # 5. GEMINI
        # ====================================================

        (
            word,
            confidence,
            status
        ) = await recognize_sign(

            image_bytes,

            mime_type
        )

        # ====================================================
        # 6. NO PREDICTION
        # ====================================================

        if word is None:

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

            print("=" * 70)

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
        # 7. SUCCESS
        # ====================================================

        print(
            "FINAL WORD:",
            word
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
                word,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "status":
                "Prediction successful"
        }

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
# Start command:
#
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# ============================================================