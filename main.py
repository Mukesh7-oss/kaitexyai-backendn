# ============================================================
# KAITEXY AI
# GEMINI VISION 10-CLASS ASL SIGN LANGUAGE BACKEND
# ============================================================
#
# Flutter
#     |
#     | POST /predict-sign
#     | multipart/form-data
#     | field = file
#     v
# FastAPI
#     |
#     v
# Gemini Vision
#     |
#     v
# EXACTLY 10 CLASSES + UNKNOWN
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
# NO LANDMARKS
# NO GENERAL WORD RECOGNITION
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
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "Kaitexy AI Gemini 10-Class Sign Recognition Backend"

APP_VERSION = "10.0-GEMINI-10CLASS"

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
# EXACT SUPPORTED CLASSES
# ============================================================

SUPPORTED_CLASSES = {
    "hello",
    "please",
    "yes",
    "thank you",
    "sorry",
    "no",
    "i love you",
    "help",
    "good",
    "bye"
}


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
    print("INITIALIZING KAITEXY AI GEMINI")
    print("=" * 70)

    if not GEMINI_API_KEY:

        print("ERROR: GEMINI_API_KEY is not configured.")

        gemini_client = None
        GEMINI_READY = False

        return False

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        GEMINI_READY = True

        print("Gemini initialized successfully.")
        print(f"Gemini model: {GEMINI_MODEL}")

        print()
        print("SUPPORTED CLASSES:")

        for item in SUPPORTED_CLASSES:
            print(f"  - {item}")

        print("=" * 70)

        return True

    except Exception as error:

        print("Gemini initialization failed:")
        print(repr(error))

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
        "Kaitexy AI closed-set 10-class "
        "ASL sign recognition using Gemini Vision."
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
# GEMINI RESPONSE SCHEMA
# ============================================================

class SignResult(BaseModel):

    prediction: str = Field(
        description=(
            "Exactly one of: "
            "hello, please, yes, thank you, sorry, no, "
            "i love you, help, good, bye, UNKNOWN"
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Visual confidence from 0.0 to 1.0."
        )
    )


# ============================================================
# GEMINI SYSTEM PROMPT
# ============================================================

SIGN_PROMPT = r"""

SYSTEM ROLE

You are Kaitexy AI's dedicated visual
10-class American Sign Language recognition engine.

You are NOT a conversational assistant.

You are NOT a general image captioning system.

You MUST perform CLOSED-SET CLASSIFICATION.

Your task is to identify the visible hand-sign
from the supplied image.

============================================================
ABSOLUTE CLASS RESTRICTION
============================================================

You may ONLY return one of these exact values:

hello
please
yes
thank you
sorry
no
i love you
help
good
bye
UNKNOWN

There are exactly 10 supported sign classes.

UNKNOWN is the rejection class.

NEVER output:

hi
thanks
love
goodbye
okay
stop
water
welcome
come
go
etc.

Even if another word appears visually plausible,
it is NOT an allowed class.

============================================================
CORE RULE
============================================================

Classify the VISIBLE HAND CONFIGURATION.

Do not infer the meaning from:

- clothing
- background
- facial expression
- objects
- assumptions
- context
- what the person might be saying

Use visual evidence from the sign.

Analyze the complete visible configuration.

============================================================
VISUAL FEATURES
============================================================

Carefully inspect:

1. Number of visible hands.
2. Handshape.
3. Finger configuration.
4. Thumb configuration.
5. Palm orientation.
6. Wrist orientation.
7. Hand orientation.
8. Hand position relative to face.
9. Hand position relative to chest.
10. Contact between hands.
11. Contact with face or chest.
12. Relative position of both hands.
13. Visible movement stage, if inferable.
14. Overall gesture configuration.

Never classify using only one finger.

============================================================
SUPPORTED CLASS 1 — HELLO
============================================================

Typical ASL greeting.

Look for an open/flat hand near the
forehead/temple area with an outward-facing
orientation and a greeting/salute-like configuration.

Do NOT classify every open hand near the face
as hello.

============================================================
SUPPORTED CLASS 2 — PLEASE
============================================================

Typical ASL "please".

Look for an open/flat hand positioned
against or near the upper chest.

The configuration is associated with
the chest-rubbing/circular movement.

In a single frame, use the visible handshape
and chest relationship.

============================================================
SUPPORTED CLASS 3 — YES
============================================================

Typical ASL "yes".

Look for a closed fist / S-like handshape.

The thumb is positioned across or near
the curled fingers.

Do NOT classify every fist as yes.

============================================================
SUPPORTED CLASS 4 — THANK YOU
============================================================

Typical ASL "thank you".

Look for a flat/open hand near the
chin or mouth area.

The gesture is normally associated with
moving the hand away from the face.

Use the exact position and orientation.

============================================================
SUPPORTED CLASS 5 — SORRY
============================================================

Typical ASL "sorry".

Look for an A-like closed hand/fist
with the thumb positioned along the
fingers.

The hand is generally associated with
the chest.

Do NOT classify every fist near the chest
as sorry.

============================================================
SUPPORTED CLASS 6 — NO
============================================================

Typical ASL "no".

Look for the characteristic configuration
involving the index finger, middle finger,
and thumb.

The index and middle fingers work with
the thumb in a closing/pinching configuration.

Do NOT classify an arbitrary two-finger
gesture as no.

============================================================
SUPPORTED CLASS 7 — I LOVE YOU
============================================================

Typical ASL ILY handshape.

Three digits must be extended simultaneously:

- thumb
- index finger
- pinky finger

Middle and ring fingers should be curled.

The simultaneous thumb + index + pinky
configuration is critical.

Do NOT confuse with:

- pointing
- rock-and-roll
- three-finger gestures
- open hand

============================================================
SUPPORTED CLASS 8 — HELP
============================================================

Typical ASL "help".

Usually two hands are visible.

One hand forms an A-like/fist configuration.

The other hand is open/flat and supports
the first hand from underneath.

The relationship between the two hands
is critical.

If the required two-hand relationship
cannot be determined, prefer UNKNOWN.

============================================================
SUPPORTED CLASS 9 — GOOD
============================================================

Typical ASL "good".

Usually an open/flat hand begins near
the mouth/chin area and moves downward
toward the other hand or lower space.

Use the handshape and spatial configuration.

Do NOT classify every open hand near the
chin as good.

============================================================
SUPPORTED CLASS 10 — BYE
============================================================

Typical ASL farewell gesture.

Usually an open palm faces outward.

The fingers may be in a waving/bending
configuration.

A single photograph may capture one stage
of the waving movement.

Do NOT classify every open outward palm
as bye.

============================================================
CRITICAL CLASS DIFFERENTIATION
============================================================

HELLO vs THANK YOU

Both may involve an open hand near the face.

HELLO:
forehead/temple region.

THANK YOU:
chin/mouth region.

Use the exact spatial position.

------------------------------------------------------------

PLEASE vs SORRY

PLEASE:
open/flat hand near chest.

SORRY:
closed/A-like hand near chest.

Handshape is extremely important.

------------------------------------------------------------

THANK YOU vs GOOD

Both may involve an open hand near the
chin/mouth.

Carefully examine the hand's position
and apparent direction.

------------------------------------------------------------

BYE vs HELLO

Both may involve an open hand.

HELLO:
greeting configuration near forehead/temple.

BYE:
outward-facing farewell/waving configuration.

------------------------------------------------------------

YES vs SORRY

Both may use fist-like handshapes.

YES:
S-like fist configuration.

SORRY:
A-like handshape associated with chest.

------------------------------------------------------------

NO vs I LOVE YOU

NO:
index + middle + thumb closing configuration.

I LOVE YOU:
thumb + index + pinky extended,
middle + ring curled.

------------------------------------------------------------

HELP

Require the characteristic relationship
between two hands.

============================================================
CAMERA MIRRORING
============================================================

The image may be horizontally mirrored.

Do NOT reject a valid sign because it is
performed with the opposite hand.

Left-hand and right-hand versions should
be treated as equivalent where appropriate.

============================================================
IMAGE QUALITY
============================================================

First determine whether the sign is
visually recognizable.

Return UNKNOWN when:

- hand is not visible
- important fingers are cropped
- image is severely blurry
- hand is obstructed
- required second hand is missing
- lighting prevents recognition
- gesture is ambiguous
- multiple classes are equally plausible
- gesture does not belong to the supported classes

============================================================
CLOSED-SET DECISION PROCESS
============================================================

Internally perform:

STEP 1:
Count visible hands.

STEP 2:
Identify handshape.

STEP 3:
Identify finger configuration.

STEP 4:
Identify thumb configuration.

STEP 5:
Identify palm orientation.

STEP 6:
Identify hand position relative to face/chest.

STEP 7:
Analyze both-hand relationship when applicable.

STEP 8:
Compare against ALL 10 supported classes.

STEP 9:
Eliminate visually inconsistent classes.

STEP 10:
Select the strongest visually supported class.

STEP 11:
If evidence is insufficient, return UNKNOWN.

============================================================
ANTI-GUESSING
============================================================

UNKNOWN is a valid answer.

Do NOT force a prediction.

Incorrect prediction is worse than UNKNOWN.

Only return a supported class when the
visible evidence is sufficiently strong.

============================================================
CONFIDENCE
============================================================

Confidence must represent visual certainty.

0.90–1.00:
Very strong visual match.

0.80–0.89:
Strong visual match.

0.70–0.79:
Reasonably strong match.

0.50–0.69:
Weak/uncertain evidence.

0.00–0.49:
Very uncertain.

For UNKNOWN, confidence should normally
be below 0.70.

Do NOT artificially increase confidence.

============================================================
FINAL OUTPUT
============================================================

Return ONLY valid JSON.

Required structure:

{
  "prediction": "hello",
  "confidence": 0.94
}

The prediction MUST be exactly one of:

hello
please
yes
thank you
sorry
no
i love you
help
good
bye
UNKNOWN

No explanation.

No Markdown.

No extra fields.

============================================================
FINAL RULE
============================================================

CLOSED SET.

10 CLASSES + UNKNOWN.

NEVER OUTPUT ANOTHER CLASS.

NEVER GUESS WHEN VISUAL EVIDENCE IS INSUFFICIENT.

CLASSIFY THE COMPLETE VISIBLE GESTURE.

"""


# ============================================================
# IMAGE TYPES
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
# NORMALIZE + VALIDATE PREDICTION
# ============================================================

def normalize_prediction(
    prediction,
    confidence
):

    if not isinstance(prediction, str):

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

    normalized = prediction.lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # HARD CLOSED-SET VALIDATION
    # --------------------------------------------------------

    if normalized not in SUPPORTED_CLASSES:

        print(
            "REJECTED INVALID GEMINI CLASS:",
            repr(normalized)
        )

        return None, 0.0

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if score < CONFIDENCE_THRESHOLD:

        return None, score

    return normalized, score


# ============================================================
# GEMINI RECOGNITION
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
        print("GEMINI RESPONSE")
        print("-" * 70)

        raw_text = getattr(
            response,
            "text",
            ""
        ) or ""

        print(
            "Raw Gemini response:",
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

            print(
                "Gemini prediction:",
                prediction
            )

            print(
                "Gemini confidence:",
                confidence
            )

            word, score = normalize_prediction(
                prediction,
                confidence
            )

            if word is None:

                return (
                    None,
                    score,
                    "Unknown or low-confidence sign"
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

                print(
                    "Fallback prediction:",
                    prediction
                )

                print(
                    "Fallback confidence:",
                    confidence
                )

                word, score = normalize_prediction(
                    prediction,
                    confidence
                )

                if word is None:

                    return (
                        None,
                        score,
                        "Unknown or low-confidence sign"
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
            "Kaitexy AI Gemini 10-Class Backend is running.",

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

        "recognition_level":
            "10-class closed-set",

        "supported_classes":
            sorted(
                list(SUPPORTED_CLASSES)
            ),

        "fixed_vocabulary":
            True
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
            "10-class closed-set",

        "supported_classes":
            sorted(
                list(SUPPORTED_CLASSES)
            ),

        "fixed_vocabulary":
            True,

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
    print("NEW SIGN PREDICTION REQUEST")
    print("=" * 70)

    try:

        # ====================================================
        # GEMINI CHECK
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
        # MIME TYPE
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
        # READ IMAGE
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
        # IMAGE SIZE
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
        # GEMINI
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
        # NO ACCEPTED PREDICTION
        # ====================================================

        if word is None:

            print(
                "FINAL PREDICTION: UNKNOWN"
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

                "prediction":
                    "UNKNOWN",

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "status":
                    status
            }

        # ====================================================
        # SUCCESS
        # ====================================================

        print(
            "FINAL PREDICTION:",
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

                "prediction":
                    "UNKNOWN",

                "confidence":
                    0.0,

                "status":
                    "Server error",

                "error":
                    str(error)
            }
        )


# ============================================================
# RENDER START COMMAND
# ============================================================
#
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# ============================================================