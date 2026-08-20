# ============================================================
# KAITEXY AI - GEMINI SIGN LANGUAGE BACKEND
# ============================================================
#
# Flutter
#    ↓
# FastAPI
#    ↓
# Gemini Vision
#    ↓
# Sign prediction
#    ↓
# Flutter builds text
#    ↓
# /correct-text
#    ↓
# Gemini text correction
#
# ============================================================

import asyncio
import os
import re
from typing import Optional, Literal

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Kaitexy AI Sign Language Backend"
APP_VERSION = "7.2-GEMINI"

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CONFIDENCE_THRESHOLD",
        "0.60"
    )
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client: Optional[genai.Client] = None

GEMINI_READY = False


def initialize_gemini() -> bool:

    global gemini_client
    global GEMINI_READY

    try:

        if not GEMINI_API_KEY:

            print(
                "ERROR: GEMINI_API_KEY environment "
                "variable is missing."
            )

            GEMINI_READY = False

            return False

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        GEMINI_READY = True

        print(
            f"Gemini initialized successfully: "
            f"{GEMINI_MODEL}"
        )

        return True

    except Exception as error:

        print(
            "Gemini initialization error:",
            repr(error)
        )

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
        "AI-powered sign language recognition "
        "backend using Gemini Vision."
    ),
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
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 60)
    print("Starting Kaitexy AI Gemini Backend")
    print("=" * 60)

    initialize_gemini()

    print("=" * 60)

    if GEMINI_READY:
        print(
            "Kaitexy AI Gemini Backend READY"
        )
    else:
        print(
            "WARNING: Gemini is NOT ready"
        )

    print("=" * 60)


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
# GEMINI RESPONSE MODEL
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
            "One uppercase ASL alphabet letter "
            "from A to Z, or UNKNOWN."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Recognition confidence from "
            "0.0 to 1.0."
        )
    )


# ============================================================
# SIGN RECOGNITION PROMPT
# ============================================================

SIGN_PROMPT = """
You are the vision recognition engine for Kaitexy AI.

Your ONLY task is to recognize ONE STATIC ASL
FINGERSPELLING ALPHABET LETTER from the provided image.

Allowed letters:

A B C D E F G H I J K L M
N O P Q R S T U V W X Y Z

Analyze ONLY the visible hand and fingers.

IMPORTANT RULES:

1. Identify the hand shape carefully.
2. Pay attention to finger position.
3. Pay attention to thumb position.
4. Pay attention to finger separation.
5. Pay attention to palm orientation.
6. Ignore the person's face.
7. Ignore clothing.
8. Ignore the background.
9. Ignore written text.
10. Ignore objects that are not part of the hand.
11. Do not return a word.
12. Do not return multiple letters.
13. Return exactly ONE letter when sufficiently clear.
14. If the hand is not visible, badly cropped, or genuinely
    impossible to identify, return UNKNOWN.
15. Do not invent a letter.
16. This is STATIC ASL FINGERSPELLING.

Confidence must be between 0.0 and 1.0.

Return ONLY the structured response requested by the API.
"""


# ============================================================
# NORMALIZE PREDICTION
# ============================================================

def normalize_prediction(
    prediction: object,
    confidence: object
):

    if not isinstance(
        prediction,
        str
    ):
        return None, 0.0

    prediction = prediction.strip().upper()

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

    if prediction == "UNKNOWN":

        return None, score

    prediction = re.sub(
        r"[^A-Z]",
        "",
        prediction
    )

    if (
        len(prediction) != 1
        or prediction not in LABELS
    ):

        return None, score

    return prediction, score


# ============================================================
# GEMINI SIGN RECOGNITION
# ============================================================

async def recognize_sign(
    image_bytes: bytes,
    mime_type: str
):

    if not GEMINI_READY:

        return (
            None,
            0.0,
            "Gemini unavailable"
        )

    try:

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        # ----------------------------------------------------
        # Gemini request
        # ----------------------------------------------------

        response = await asyncio.to_thread(

            gemini_client.models.generate_content,

            model=GEMINI_MODEL,

            contents=[
                image_part,
                SIGN_PROMPT
            ],

            config=types.GenerateContentConfig(

                # Structured JSON
                response_mime_type="application/json",

                response_schema=SignResult,

                # Give the model enough room.
                max_output_tokens=100,

                # Keep reasoning minimal for speed.
                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                ),

                # Prevent tools / automatic function behavior.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print("=" * 60)
        print("GEMINI RESPONSE RECEIVED")
        print("=" * 60)

        raw_text = getattr(
            response,
            "text",
            ""
        ) or ""

        print(
            "RAW TEXT:",
            repr(raw_text)
        )

        # ----------------------------------------------------
        # BEST METHOD:
        # Use SDK parsed structured response.
        # ----------------------------------------------------

        parsed = getattr(
            response,
            "parsed",
            None
        )

        if parsed is not None:

            print(
                "PARSED RESPONSE:",
                repr(parsed)
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

            if letter is None:

                return (
                    None,
                    score,
                    "Gesture not confidently recognized"
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

        # ----------------------------------------------------
        # FALLBACK:
        # Parse raw JSON if SDK did not populate .parsed.
        # ----------------------------------------------------

        if raw_text:

            import json

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

                if letter is None:

                    return (
                        None,
                        score,
                        "Gesture not confidently recognized"
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

            except Exception as parse_error:

                print(
                    "JSON parsing error:",
                    repr(parse_error)
                )

        # ----------------------------------------------------
        # Empty / incomplete response.
        # ----------------------------------------------------

        print(
            "Gemini returned no complete structured response."
        )

        return (
            None,
            0.0,
            "Invalid Gemini response"
        )

    except Exception as error:

        print("=" * 60)
        print(
            "GEMINI SIGN RECOGNITION ERROR"
        )
        print("=" * 60)

        print(
            repr(error)
        )

        print("=" * 60)

        return (
            None,
            0.0,
            "Gemini recognition failed"
        )


# ============================================================
# TEXT CORRECTION PROMPT
# ============================================================

TEXT_CORRECTION_PROMPT = """
You are the language correction engine for Kaitexy AI.

Correct the spelling and grammar of the supplied sentence.

Rules:

1. Preserve the original meaning.
2. Do not add information.
3. Do not remove important information.
4. Do not invent words.
5. Keep the sentence natural.
6. Return ONLY the corrected sentence.
7. Do not explain your changes.

Text:
"""


# ============================================================
# GEMINI TEXT CORRECTION
# ============================================================

async def correct_text_gemini(
    text: str
) -> str:

    if not text:
        return ""

    if not GEMINI_READY:
        return text

    try:

        prompt = (
            TEXT_CORRECTION_PROMPT
            + text.strip()
        )

        response = await asyncio.to_thread(

            gemini_client.models.generate_content,

            model=GEMINI_MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                max_output_tokens=100,

                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                ),

                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )
        )

        corrected = getattr(
            response,
            "text",
            ""
        ) or ""

        corrected = corrected.strip()

        if not corrected:
            return text

        if (
            len(corrected) >= 2
            and corrected[0] == '"'
            and corrected[-1] == '"'
        ):

            corrected = (
                corrected[1:-1]
                .strip()
            )

        return corrected or text

    except Exception as error:

        print(
            "Gemini text correction error:",
            repr(error)
        )

        return text


# ============================================================
# REQUEST MODEL
# ============================================================

class CorrectTextRequest(BaseModel):

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

        "labels":
            len(LABELS),

        "input_size":
            63,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "device":
            "gemini-cloud",

        "sign_recognition":
            "Gemini Vision",

        "text_correction":
            "Gemini"

    }


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

        "health":
            "/health",

        "prediction_endpoint":
            "/predict-sign",

        "text_endpoint":
            "/correct-text"

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
        # Gemini availability
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Validate image type
        # ----------------------------------------------------

        content_type = (
            file.content_type or ""
        ).lower()

        mime_map = {

            "image/jpeg":
                "image/jpeg",

            "image/jpg":
                "image/jpeg",

            "image/png":
                "image/png",

            "image/webp":
                "image/webp"

        }

        if content_type not in mime_map:

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

        image_bytes = await file.read()

        if not image_bytes:

            return {

                "prediction": "",
                "confidence": 0.0,
                "status":
                    "Empty image"

            }

        # ----------------------------------------------------
        # File size protection
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Gemini recognition
        # ----------------------------------------------------

        (
            letter,
            confidence,
            status
        ) = await recognize_sign(

            image_bytes,

            mime_map[
                content_type
            ]

        )

        # ----------------------------------------------------
        # No prediction
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
                    status

            }

        # ----------------------------------------------------
        # Successful prediction
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
            "Predict endpoint error:",
            repr(error)
        )

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

    if not GEMINI_READY:

        return {

            "raw_text":
                raw_text,

            "corrected_text":
                raw_text,

            "status":
                "Gemini unavailable; "
                "original text returned"

        }

    try:

        corrected = (
            await correct_text_gemini(
                raw_text
            )
        )

        return {

            "raw_text":
                raw_text,

            "corrected_text":
                corrected,

            "status":
                "Text corrected successfully"

        }

    except Exception as error:

        print(
            "Text correction endpoint error:",
            repr(error)
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
# RENDER START COMMAND
# ============================================================
#
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# ============================================================