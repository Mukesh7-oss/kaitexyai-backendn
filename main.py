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

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Kaitexy AI Gemini 10-Class Backend"
APP_VERSION = "11.0-GEMINI-10CLASS-FIXED"

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
        "0.60"
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

        print()
        print("Supported classes:")

        for item in sorted(SUPPORTED_CLASSES):
            print(f"  - {item}")

        print()
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

def normalize_mime_type(
    content_type: str
) -> Optional[str]:

    if not content_type:
        return None

    content_type = (
        content_type
        .lower()
        .strip()
    )

    # Remove optional parameters
    content_type = (
        content_type
        .split(";")[0]
        .strip()
    )

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

    # --------------------------------------------------------
    # Prediction must be a string
    # --------------------------------------------------------

    if not isinstance(
        prediction,
        str
    ):

        return None, 0.0

    prediction = (
        prediction
        .strip()
        .lower()
    )

    # Normalize whitespace

    prediction = re.sub(
        r"\s+",
        " ",
        prediction
    ).strip()

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    try:

        score = float(
            confidence
        )

    except Exception:

        score = 0.0

    # --------------------------------------------------------
    # Handle percentages accidentally returned
    # --------------------------------------------------------

    if score > 1.0 and score <= 100.0:

        score = score / 100.0

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

    if prediction == "unknown":

        return None, score

    # --------------------------------------------------------
    # CLOSED SET
    # --------------------------------------------------------

    if prediction not in SUPPORTED_CLASSES:

        print(
            f"REJECTED GEMINI CLASS: {prediction}"
        )

        return None, score

    return prediction, score


# ============================================================
# EXTRACT JSON FROM TEXT
# ============================================================

def extract_json_object(
    text: str
):

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict
        ):

            return parsed

    except Exception:
        pass

    # --------------------------------------------------------
    # Find JSON object embedded in text
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                dict
            ):

                return parsed

        except Exception:
            pass

    return None


# ============================================================
# PARSE GEMINI RESPONSE
# ============================================================

def parse_gemini_response(
    response
):

    print()
    print("-" * 70)
    print("GEMINI RESPONSE")
    print("-" * 70)

    # ========================================================
    # 1. TRY PARSED RESPONSE
    # ========================================================

    parsed_object = getattr(
        response,
        "parsed",
        None
    )

    print(
        "Parsed object:",
        repr(parsed_object)
    )

    if parsed_object is not None:

        try:

            # Pydantic-like object

            if hasattr(
                parsed_object,
                "model_dump"
            ):

                parsed_object = (
                    parsed_object
                    .model_dump()
                )

            # Dictionary

            if isinstance(
                parsed_object,
                dict
            ):

                prediction = (
                    parsed_object
                    .get(
                        "prediction",
                        ""
                    )
                )

                confidence = (
                    parsed_object
                    .get(
                        "confidence",
                        0.0
                    )
                )

                word, score = (
                    normalize_prediction(
                        prediction,
                        confidence
                    )
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
                "Parsed-object processing error:",
                repr(error)
            )

    # ========================================================
    # 2. RAW TEXT
    # ========================================================

    raw_text = getattr(
        response,
        "text",
        None
    )

    print(
        "Raw response:",
        repr(raw_text)
    )

    if not raw_text:

        print(
            "Gemini returned empty text."
        )

        return (
            None,
            0.0,
            "Empty Gemini response"
        )

    cleaned = raw_text.strip()

    print(
        "Cleaned response:",
        repr(cleaned)
    )

    # ========================================================
    # 3. PARSE JSON
    # ========================================================

    data = extract_json_object(
        cleaned
    )

    if data is None:

        print(
            "Direct JSON parsing failed."
        )

        # ----------------------------------------------------
        # Emergency fallback
        #
        # If Gemini ignores JSON formatting and simply returns
        # one of the allowed class names, accept that safely.
        # ----------------------------------------------------

        fallback = (
            cleaned
            .strip()
            .lower()
        )

        fallback = re.sub(
            r"[^a-zA-Z\s]",
            "",
            fallback
        )

        fallback = re.sub(
            r"\s+",
            " ",
            fallback
        ).strip()

        print(
            "Fallback text:",
            repr(fallback)
        )

        if fallback in SUPPORTED_CLASSES:

            print(
                "Accepted direct class-name fallback."
            )

            return (
                fallback,
                0.80,
                "Prediction successful"
            )

        print(
            "Gemini response could not be parsed."
        )

        return (
            None,
            0.0,
            "Invalid Gemini response"
        )

    # ========================================================
    # 4. EXTRACT FIELDS
    # ========================================================

    print(
        "Parsed JSON:",
        repr(data)
    )

    prediction = data.get(
        "prediction",
        ""
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

    # ========================================================
    # 5. NORMALIZE
    # ========================================================

    word, score = normalize_prediction(
        prediction,
        confidence
    )

    # ========================================================
    # 6. UNKNOWN
    # ========================================================

    if word is None:

        return (
            None,
            score,
            "Sign not confidently recognized"
        )

    # ========================================================
    # 7. CONFIDENCE THRESHOLD
    # ========================================================

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

    # ========================================================
    # SUCCESS
    # ========================================================

    return (
        word,
        score,
        "Prediction successful"
    )


# ============================================================
# SIGN CLASSIFICATION PROMPT
# ============================================================

SIGN_PROMPT = """
You are Kaitexy AI, a closed-set visual sign classifier.

Look ONLY at the uploaded image.

Your task is to identify which ONE of these 10 signs is
visually represented:

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

If the image does not provide enough visual evidence,
return UNKNOWN.

IMPORTANT:

Do not explain your answer.

Do not describe the image.

Do not write a sentence.

Do not say "The sign is".

Return ONLY the structured result.

VISUAL RULES:

HELLO:
Open hand near forehead/temple in a greeting/salute-like
configuration.

PLEASE:
Open or flat hand positioned against or near the upper chest.

YES:
Closed fist / S-handshape with thumb positioned over the
curled fingers.

THANK YOU:
Open/flat hand near the chin or mouth, associated with the
thank-you gesture.

SORRY:
A-like closed handshape positioned against or near the chest.

NO:
Index and middle fingers interacting with the thumb in the
characteristic closing/pinching configuration.

I LOVE YOU:
Thumb, index finger and pinky extended while middle and
ring fingers are curled.

HELP:
Two-hand configuration. One hand supports an A-like/fist
configuration from underneath.

GOOD:
Open/flat hand near the chin/mouth with the characteristic
good-sign configuration.

BYE:
Open hand with palm outward in a farewell/waving
configuration.

DISAMBIGUATION:

hello vs thank you:
hello is associated with forehead/temple.
thank you is associated with chin/mouth.

please vs sorry:
please generally uses an open/flat hand.
sorry generally uses an A-like closed hand.

yes vs sorry:
use both handshape and body location.

no:
require the characteristic finger/thumb interaction.

i love you:
require thumb + index + pinky extended together.

help:
require evidence of two hands.

good vs thank you:
carefully examine the hand position and configuration.

bye vs hello:
use the hand location and farewell/greeting configuration.

UNKNOWN if:

- no hand is visible
- the hand is severely cropped
- fingers cannot be distinguished
- image is severely blurred
- hand is obstructed
- required second hand is missing
- gesture is ambiguous
- multiple classes are equally plausible
- there is insufficient visual evidence

Do NOT guess.

Confidence must be between 0.0 and 1.0.

Confidence represents visual certainty.

Return a valid structured response with:

prediction
confidence
"""


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

        print(
            "Gemini model:",
            GEMINI_MODEL
        )

        # ====================================================
        # IMAGE PART
        # ====================================================

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        # ====================================================
        # STRUCTURED OUTPUT SCHEMA
        # ====================================================

        response_schema = {
            "type": "OBJECT",
            "properties": {

                "prediction": {
                    "type": "STRING",
                    "enum": [
                        "hello",
                        "please",
                        "yes",
                        "thank you",
                        "sorry",
                        "no",
                        "i love you",
                        "help",
                        "good",
                        "bye",
                        "UNKNOWN"
                    ]
                },

                "confidence": {
                    "type": "NUMBER"
                }
            },

            "required": [
                "prediction",
                "confidence"
            ]
        }

        # ====================================================
        # GEMINI REQUEST
        # ====================================================

        response = await asyncio.to_thread(

            gemini_client.models.generate_content,

            model=GEMINI_MODEL,

            contents=[
                SIGN_PROMPT,
                image_part
            ],

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema=response_schema,

                # Gemini 3.x:
                # minimal thinking gives faster response
                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                ),

                # Enough room for tiny JSON output
                max_output_tokens=256
            )
        )

        print(
            "Gemini request completed."
        )

        # ====================================================
        # PARSE
        # ====================================================

        result = parse_gemini_response(
            response
        )

        print(
            "=" * 70
        )

        return result

    except Exception as error:

        print()
        print("=" * 70)
        print("GEMINI API ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(error).__name__
        )

        print(
            "Error:",
            repr(error)
        )

        print(
            "=" * 70
        )

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
        # 6. UNKNOWN
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

            print(
                "=" * 70
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
        # 7. SUCCESS
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

        print(
            "=" * 70
        )

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
            "Error type:",
            type(error).__name__
        )

        print(
            "Error:",
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

# uvicorn main:app --host 0.0.0.0 --port 8000


# ============================================================
# RENDER
# ============================================================

# uvicorn main:app --host 0.0.0.0 --port $PORT