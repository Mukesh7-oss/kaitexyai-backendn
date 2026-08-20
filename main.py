# ============================================================
# KAITEXY AI
# GEMINI VISION 10-CLASS SIGN LANGUAGE BACKEND
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
# CLOSED-SET 10-CLASS SIGN RECOGNITION
#
# Supported:
#   hello
#   please
#   yes
#   thank you
#   sorry
#   no
#   i love you
#   help
#   good
#   bye
#
# NO PyTorch
# NO MediaPipe
# NO .pt MODEL
# ============================================================

import asyncio
import json
import os
import re
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Kaitexy AI Gemini 10-Class Backend"
APP_VERSION = "10.1-GEMINI-10CLASS"

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
# EXACT CLOSED VOCABULARY
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
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Kaitexy AI closed-set 10-class "
        "sign-language recognition using Gemini Vision."
    )
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)


# ============================================================
# GEMINI INITIALIZATION
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

        print("Supported classes:")

        for item in sorted(SUPPORTED_CLASSES):
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
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print()
    print("=" * 70)
    print("STARTING KAITEXY AI BACKEND")
    print("=" * 70)

    initialize_gemini()

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

    prediction: str
    confidence: float


# ============================================================
# MASTER SIGN RECOGNITION PROMPT
# ============================================================

SIGN_PROMPT = r"""
SYSTEM ROLE:

You are Kaitexy AI's dedicated visual sign-language
classification engine.

You are NOT a conversational assistant.

You are NOT an image captioning system.

You are NOT allowed to generate arbitrary words.

Your ONLY task is closed-set visual classification.

============================================================
SUPPORTED CLASSES
============================================================

There are EXACTLY 10 supported signs:

1. hello
2. please
3. yes
4. thank you
5. sorry
6. no
7. i love you
8. help
9. good
10. bye

There is also one rejection result:

UNKNOWN

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

Never output any other word.

============================================================
IMPORTANT
============================================================

This is a SINGLE IMAGE classification task.

You must classify the visible hand configuration.

Do not rely on:

- clothing
- background
- objects
- face identity
- assumed situation
- imagined motion
- captions
- common sense
- conversational context

Use the visible sign configuration.

============================================================
VISUAL ANALYSIS
============================================================

Analyze the complete visible gesture.

Consider:

1. Number of hands.
2. Handshape.
3. Finger configuration.
4. Thumb configuration.
5. Palm orientation.
6. Wrist orientation.
7. Hand orientation.
8. Hand position relative to the face.
9. Hand position relative to the chest.
10. Relationship between both hands.
11. Contact between hands.
12. Contact with face or chest.
13. Overall geometric configuration.

Do NOT classify from a single finger.

Do NOT classify from the general appearance of the hand alone.

============================================================
CLASS DEFINITIONS
============================================================

HELLO:

Typically an open hand near the forehead/temple,
associated with a greeting/salute-like configuration.

Look for an open hand and location near the forehead.

Do not classify every open hand near the face as hello.

------------------------------------------------------------

PLEASE:

Typically an open/flat hand positioned against or
near the upper chest.

The characteristic configuration is associated with
the ASL "please" sign.

Look for an open hand and chest location.

------------------------------------------------------------

YES:

Typically a closed fist / S-handshape.

The thumb is positioned over or across the curled fingers.

Do not classify every fist as yes.

------------------------------------------------------------

THANK YOU:

Typically an open/flat hand beginning near the chin
or mouth and moving outward.

In one image, examine the characteristic hand position
near the chin/mouth.

Do not classify every hand near the face as thank you.

------------------------------------------------------------

SORRY:

Typically an A-like closed handshape positioned
against or near the chest.

The handshape is important.

Do not classify every fist near the chest as sorry.

------------------------------------------------------------

NO:

Typically the index and middle fingers interact with
the thumb in a closing/pinching configuration.

The complete three-part configuration matters.

Do not classify an arbitrary two-finger gesture as no.

------------------------------------------------------------

I LOVE YOU:

The classic ILY handshape:

- thumb extended
- index finger extended
- pinky extended
- middle finger curled
- ring finger curled

The simultaneous configuration is essential.

------------------------------------------------------------

HELP:

Typically requires TWO hands.

One hand forms an A-like/fist configuration.

The other hand supports it from underneath.

The relationship between both hands is essential.

If the second hand is missing or cannot be determined,
prefer UNKNOWN.

------------------------------------------------------------

GOOD:

Typically uses an open/flat hand near the chin/mouth
with the gesture moving downward toward the other hand
or lower neutral space.

Distinguish carefully from thank you.

------------------------------------------------------------

BYE:

Typically an open hand with the palm facing outward,
associated with a waving/farewell configuration.

Do not classify every open palm as bye.

============================================================
SIMILAR CLASS DISAMBIGUATION
============================================================

HELLO vs THANK YOU:

Look at exact location.

Hello:
forehead/temple region.

Thank you:
chin/mouth region.

PLEASE vs SORRY:

Please:
open/flat hand.

Sorry:
closed/A-like hand.

YES vs SORRY:

Both can appear fist-like.

Use both handshape and body location.

NO:

Require the characteristic index/middle/thumb interaction.

I LOVE YOU:

Require thumb + index + pinky extended while
middle + ring are curled.

HELP:

Require evidence of the two-hand relationship.

GOOD vs THANK YOU:

Both can involve the face/chin region.

Use the spatial configuration and visible orientation.

BYE vs HELLO:

Both can involve an open hand.

Use hand location and farewell/greeting configuration.

============================================================
CAMERA VARIATION
============================================================

Allow reasonable variation caused by:

- left/right hand
- camera mirroring
- wrist rotation
- camera angle
- perspective
- lighting
- skin tone
- distance
- signer variation

Do not require pixel-perfect matching.

However, variation must still preserve the underlying
hand configuration.

============================================================
UNKNOWN RULE
============================================================

Return UNKNOWN if:

- no hand is visible
- the hand is severely cropped
- fingers cannot be distinguished
- image is severely blurred
- hand is obstructed
- required second hand is missing
- the configuration is ambiguous
- multiple classes are equally plausible
- it is an ordinary gesture
- there is insufficient visual evidence

Do NOT guess.

UNKNOWN is a valid classification.

============================================================
DECISION PROCESS
============================================================

Internally:

1. Identify visible hands.
2. Analyze handshape.
3. Analyze fingers.
4. Analyze thumb.
5. Analyze palm orientation.
6. Analyze body location.
7. Analyze hand relationships.
8. Compare against ALL 10 classes.
9. Eliminate incompatible classes.
10. Select the strongest visual match.
11. If evidence is insufficient, choose UNKNOWN.

Do not choose a class merely because it is plausible.

============================================================
CONFIDENCE
============================================================

Confidence is visual certainty.

0.90 - 1.00:
Very strong visual match.

0.80 - 0.89:
Strong visual match.

0.70 - 0.79:
Reasonably strong match.

0.50 - 0.69:
Weak/ambiguous.

0.00 - 0.49:
Very uncertain.

For UNKNOWN, confidence should normally be below 0.70.

Never artificially increase confidence.

============================================================
OUTPUT
============================================================

Return ONLY JSON.

Exactly these two fields:

{
  "prediction": "hello",
  "confidence": 0.94
}

No explanation.

No Markdown.

No additional fields.

prediction MUST be exactly:

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

============================================================
FINAL RULE
============================================================

CLOSED SET.

10 CLASSES + UNKNOWN.

NO OTHER OUTPUT IS VALID.

WHEN UNCERTAIN, RETURN UNKNOWN.
"""


# ============================================================
# SUPPORTED MIME TYPES
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
# NORMALIZE MIME TYPE
# ============================================================

def normalize_mime_type(content_type: str) -> Optional[str]:

    if not content_type:
        return None

    content_type = content_type.lower().strip()

    # Remove parameters such as:
    # image/jpeg; charset=utf-8

    content_type = content_type.split(";")[0].strip()

    return SUPPORTED_MIME_TYPES.get(
        content_type
    )


# ============================================================
# NORMALIZE PREDICTION
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

    # Normalize spacing/case

    prediction = re.sub(
        r"\s+",
        " ",
        prediction
    ).strip().lower()

    # Exact UNKNOWN handling

    if prediction == "unknown":
        return None, score

    # Exact closed-set validation

    if prediction not in SUPPORTED_CLASSES:
        print(
            f"REJECTED GEMINI CLASS: {prediction}"
        )

        return None, score

    return prediction, score


# ============================================================
# PARSE GEMINI JSON
# ============================================================

def parse_gemini_response(
    response
):

    raw_text = getattr(
        response,
        "text",
        None
    )

    print()
    print("-" * 70)
    print("GEMINI RESPONSE")
    print("-" * 70)

    print(
        "Raw response:",
        repr(raw_text)
    )

    if not raw_text:
        print("Gemini returned empty text.")

        return None, 0.0, "Empty Gemini response"

    # --------------------------------------------------------
    # Remove accidental markdown fences
    # --------------------------------------------------------

    cleaned = raw_text.strip()

    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    cleaned = cleaned.strip()

    print(
        "Cleaned response:",
        repr(cleaned)
    )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            cleaned
        )

    except Exception as error:

        print(
            "JSON parse failed:",
            repr(error)
        )

        # Try extracting the JSON object

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL
        )

        if not match:

            return (
                None,
                0.0,
                "Invalid Gemini response"
            )

        try:

            data = json.loads(
                match.group(0)
            )

        except Exception as second_error:

            print(
                "JSON extraction failed:",
                repr(second_error)
            )

            return (
                None,
                0.0,
                "Invalid Gemini response"
            )

    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

    prediction = data.get(
        "prediction"
    )

    confidence = data.get(
        "confidence",
        0.0
    )

    print(
        "Gemini prediction:",
        repr(prediction)
    )

    print(
        "Gemini confidence:",
        repr(confidence)
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    word, score = normalize_prediction(
        prediction,
        confidence
    )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if word is None:

        return (
            None,
            score,
            "Sign not confidently recognized"
        )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if score < CONFIDENCE_THRESHOLD:

        print(
            "Prediction rejected because confidence "
            "is below threshold."
        )

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


# ============================================================
# GEMINI VISION RECOGNITION
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

        print()
        print("=" * 70)
        print("SENDING IMAGE TO GEMINI")
        print("=" * 70)

        print(
            "Image bytes:",
            len(image_bytes)
        )

        print(
            "Image MIME:",
            mime_type
        )

        # ----------------------------------------------------
        # IMAGE PART
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

                temperature=0.0,

                max_output_tokens=100
            )
        )

        print(
            "Gemini request completed."
        )

        # ----------------------------------------------------
        # PARSE RESPONSE TEXT
        # ----------------------------------------------------

        result = parse_gemini_response(
            response
        )

        print("=" * 70)

        return result

    except Exception as error:

        print()
        print("=" * 70)
        print("GEMINI API ERROR")
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
                SUPPORTED_CLASSES
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
                SUPPORTED_CLASSES
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
        # 1. GEMINI READY
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

        original_content_type = (
            file.content_type or ""
        )

        print(
            "Received MIME:",
            original_content_type
        )

        mime_type = normalize_mime_type(
            original_content_type
        )

        if mime_type is None:

            return JSONResponse(

                status_code=400,

                content={

                    "prediction": "",

                    "confidence": 0.0,

                    "status":
                        "Unsupported image type"
                }
            )

        print(
            "Normalized MIME:",
            mime_type
        )

        # ====================================================
        # 3. READ IMAGE
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
        # 4. SIZE LIMIT
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
        # 6. NO ACCEPTED PREDICTION
        # ====================================================

        if word is None:

            print()
            print(
                "FINAL RESULT: UNKNOWN"
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
        # 7. FINAL SUCCESS
        # ====================================================

        print()
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

                "prediction": "",

                "confidence": 0.0,

                "status":
                    "Server error",

                "error":
                    str(error)
            }
        )


# ============================================================
# LOCAL RUN
# ============================================================
#
# uvicorn main:app --host 0.0.0.0 --port 8000
#
# RENDER:
#
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# ============================================================