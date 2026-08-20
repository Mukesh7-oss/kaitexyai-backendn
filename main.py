# ============================================================
# KAITEXY AI
# GEMINI VISION 10-CLASS SIGN LANGUAGE BACKEND
# ============================================================
#
# Flutter
#     |
#     | POST /predict-sign
#     | multipart/form-data
#     | file = JPEG image
#     v
# FastAPI
#     |
#     v
# Gemini Vision
#     |
#     v
# 10-CLASS CLOSED-SET CLASSIFICATION
#     |
#     v
# JSON
# {
#     "prediction": "hello",
#     "confidence": 0.94,
#     "status": "Prediction successful"
# }
#
# ============================================================
# NO PYTORCH
# NO MEDIAPIPE
# NO .PT MODEL
# NO 63 LANDMARKS
# NO A-Z CLASSIFICATION
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
# EXACT SUPPORTED CLASSES
# ============================================================

SUPPORTED_CLASSES = [
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
]

SUPPORTED_CLASSES_SET = set(
    SUPPORTED_CLASSES
)


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
        "Kaitexy AI closed-set 10-class "
        "sign language recognition using Gemini Vision."
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

    print()

    print("SUPPORTED CLASSES:")

    for index, class_name in enumerate(
        SUPPORTED_CLASSES,
        start=1
    ):

        print(
            f"{index:02d}. {class_name}"
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

    prediction: str = Field(
        description=(
            "Exactly one of the ten supported "
            "class names, or UNKNOWN."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Visual classification confidence "
            "between 0.0 and 1.0."
        )
    )


# ============================================================
# GEMINI SYSTEM PROMPT
# ============================================================

SIGN_PROMPT = r"""

SYSTEM ROLE

You are Kaitexy AI's dedicated visual
sign-language classification engine.

You are NOT a conversational assistant.

You are NOT an image captioning system.

You are NOT allowed to invent a meaning.

Your ONLY task is CLOSED-SET visual classification.

The supplied image contains a hand gesture.

Classify it into exactly ONE of the ten
supported sign-language classes below.

If the image does not provide enough visual
evidence for one of those ten classes,
return UNKNOWN.

============================================================
SUPPORTED CLASSES
============================================================

The ONLY valid classes are:

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

There are NO other classes.

Do not output:

hi
thanks
love
goodbye
okay
stop
welcome
water
eat
drink
friend
family
etc.

============================================================
ABSOLUTE CLOSED-SET RULE
============================================================

You MUST compare the image ONLY against
the ten supported classes.

Never introduce a new class.

Never substitute a synonym.

Never use contextual reasoning to invent
a class.

The final prediction must be:

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

or:

UNKNOWN

============================================================
VISUAL ANALYSIS
============================================================

Analyze the complete visible gesture.

Consider:

1. Number of hands.

2. Hand shape.

3. Finger configuration.

4. Thumb configuration.

5. Palm orientation.

6. Wrist orientation.

7. Hand orientation.

8. Hand position relative to the face.

9. Hand position relative to the chest.

10. Relationship between both hands.

11. Contact between hands.

12. Contact with face or body.

13. Overall anatomical configuration.

Do NOT classify based on one finger alone.

Do NOT classify based only on the location of the hand.

The complete gesture is more important than
any individual feature.

============================================================
SINGLE IMAGE RULE
============================================================

The input is normally one photograph.

You cannot reliably observe an entire movement
sequence from one photograph.

Therefore:

Do not invent movement.

Do not assume a movement that is not visible.

Use the visible hand configuration and spatial
relationship.

If a sign normally involves movement, classify it
only when the captured position provides sufficient
evidence.

Otherwise return UNKNOWN.

============================================================
CLASS 1 — HELLO
============================================================

Typical ASL greeting.

Look for an open, flat hand with fingers
generally extended together and the thumb
naturally positioned.

The hand is generally associated with a
greeting/salute-like configuration near the
forehead or temple.

Do not classify every open hand near the face
as hello.

============================================================
CLASS 2 — PLEASE
============================================================

Typical ASL "please".

Usually an open, flat hand is placed against
or near the upper chest.

The palm generally faces toward the signer.

The sign commonly involves rubbing/circular
movement over the chest.

In one frame, focus on:

- open hand
- chest location
- palm orientation

Do not classify an arbitrary open hand near
the chest as please.

============================================================
CLASS 3 — YES
============================================================

Typical ASL "yes".

Look for an S-like fist configuration.

The fingers are curled into a fist and the
thumb participates in the closed handshape.

Do not classify every fist as yes.

The complete handshape must support yes.

============================================================
CLASS 4 — THANK YOU
============================================================

Typical ASL "thank you".

Look for a flat/open hand near the chin or mouth.

The palm generally faces toward the signer.

The hand is associated with movement away from
the face.

Do not classify every open hand near the face
as thank you.

============================================================
CLASS 5 — SORRY
============================================================

Typical ASL "sorry".

Look for an A-like closed handshape.

The thumb is positioned along/against the
curled fingers.

The hand is generally near or against the chest.

The sign commonly involves circular movement
over the chest.

Require both:

- appropriate handshape
- appropriate body location

============================================================
CLASS 6 — NO
============================================================

Typical ASL "no".

Look for the characteristic interaction
between:

- index finger
- middle finger
- thumb

The fingers and thumb form a closing/pinching
configuration.

Do not classify every two-finger gesture as no.

============================================================
CLASS 7 — I LOVE YOU
============================================================

Typical ASL ILY handshape.

The following three digits are extended:

- thumb
- index finger
- pinky finger

While:

- middle finger is curled
- ring finger is curled

The combination is critical.

Do not confuse it with:

- pointing
- rock gesture
- ordinary three-finger gesture
- open hand

============================================================
CLASS 8 — HELP
============================================================

Typical ASL "help".

Usually involves TWO hands.

One hand forms an A-like/fist configuration.

The other hand is open/flat and supports
the first hand underneath.

The relationship between the two hands
is essential.

If the second hand is required but not visible,
prefer UNKNOWN.

============================================================
CLASS 9 — GOOD
============================================================

Typical ASL "good".

Usually involves an open/flat hand near the
mouth or chin followed by downward movement
toward the other hand or lower space.

In a single image, examine:

- open handshape
- face/chin position
- spatial relationship

Distinguish carefully from thank you.

============================================================
CLASS 10 — BYE
============================================================

Typical ASL farewell.

Usually an open hand with palm facing outward.

The fingers are extended and may appear in
a waving/bending configuration.

Do not classify every open outward-facing
palm as bye.

The overall configuration should be consistent
with a farewell gesture.

============================================================
CRITICAL CLASS DISTINCTIONS
============================================================

HELLO vs THANK YOU

Both can involve an open hand near the face.

HELLO:
Look toward forehead/temple and greeting-like
configuration.

THANK YOU:
Look toward chin/mouth and outward gesture
configuration.

PLEASE vs SORRY

PLEASE:
Open/flat hand near chest.

SORRY:
Closed/A-like hand near chest.

YES vs SORRY

YES:
Closed S-like hand.

SORRY:
A-like hand associated with chest.

NO vs TWO-FINGER GESTURES

Do not classify based only on seeing two fingers.

The thumb interaction must support NO.

I LOVE YOU vs OTHER THREE-FINGER GESTURES

Require:

thumb extended
+
index extended
+
pinky extended

while middle and ring fingers are curled.

HELP

Requires the relationship between two hands.

============================================================
CAMERA VARIATION
============================================================

Allow reasonable variation caused by:

- left/right hand
- mirrored camera
- wrist rotation
- camera angle
- distance
- lighting
- skin tone
- background
- hand size
- perspective
- minor finger-angle variation

Interpret the underlying hand configuration.

Do not require pixel-perfect orientation.

However, camera variation is NOT permission to guess.

============================================================
IMAGE QUALITY
============================================================

Return UNKNOWN if:

- no hand is visible
- hand is severely cropped
- fingers cannot be distinguished
- image is severely blurry
- hand is heavily obstructed
- required second hand is missing
- lighting prevents recognition
- multiple classes are equally plausible
- gesture is not one of the supported classes

============================================================
DECISION PROCESS
============================================================

Internally perform:

STEP 1:
Determine visible hand count.

STEP 2:
Determine handshape.

STEP 3:
Determine finger configuration.

STEP 4:
Determine thumb configuration.

STEP 5:
Determine palm orientation.

STEP 6:
Determine body/face location.

STEP 7:
Analyze both-hand relationship if applicable.

STEP 8:
Compare against ALL ten classes.

STEP 9:
Eliminate incompatible classes.

STEP 10:
Select the strongest visually supported class.

STEP 11:
If visual evidence is insufficient, select UNKNOWN.

Do NOT choose a class merely because it is plausible.

============================================================
ANTI-GUESSING
============================================================

UNKNOWN is a legitimate result.

Do not force a classification.

It is better to return UNKNOWN than an incorrect
classification.

Confidence must represent visual certainty.

============================================================
CONFIDENCE
============================================================

0.90 - 1.00

Extremely clear visual match.

0.80 - 0.89

Strong visual match.

0.70 - 0.79

Reasonably strong match.

0.50 - 0.69

Weak evidence.

0.00 - 0.49

Very uncertain.

For UNKNOWN, normally use confidence below 0.70.

Do not artificially increase confidence.

============================================================
FINAL OUTPUT
============================================================

Return ONLY valid JSON.

No explanation.

No reasoning.

No Markdown.

No additional fields.

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

============================================================
FINAL RULE
============================================================

CLASSIFY ONLY THE TEN SUPPORTED SIGNS.

NEVER OUTPUT ANOTHER CLASS.

NEVER INVENT A MEANING.

NEVER GUESS WHEN VISUAL EVIDENCE IS INSUFFICIENT.

RETURN ONE CLASS OR UNKNOWN.

"""


# ============================================================
# IMAGE MIME TYPES
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
# NORMALIZE GEMINI PREDICTION
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

    normalized = re.sub(
        r"\s+",
        " ",
        prediction.lower()
    ).strip()

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
        "none",
        ""
    }:

        return None, score

    # --------------------------------------------------------
    # STRICT CLOSED-SET VALIDATION
    # --------------------------------------------------------

    if normalized not in SUPPORTED_CLASSES_SET:

        print(
            "REJECTED GEMINI OUTPUT:",
            repr(prediction)
        )

        return None, 0.0

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

        # ====================================================
        # CREATE IMAGE PART
        # ====================================================

        image_part = types.Part.from_bytes(

            data=image_bytes,

            mime_type=mime_type

        )

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

                response_schema=SignResult,

                max_output_tokens=100,

                temperature=0.0

            )

        )

        # ====================================================
        # RAW RESPONSE
        # ====================================================

        print()
        print("-" * 70)
        print("GEMINI 10-CLASS PREDICTION")
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

        # ====================================================
        # STRUCTURED RESPONSE
        # ====================================================

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
                "Gemini prediction:",
                prediction
            )

            print(
                "Gemini confidence:",
                score
            )

            # =================================================
            # INVALID / UNKNOWN
            # =================================================

            if word is None:

                return (
                    None,
                    score,
                    "Sign not confidently recognized"
                )

            # =================================================
            # CONFIDENCE THRESHOLD
            # =================================================

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

        # ====================================================
        # FALLBACK JSON PARSING
        # ====================================================

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
            SUPPORTED_CLASSES,

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
            SUPPORTED_CLASSES,

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
    print("NEW 10-CLASS SIGN REQUEST")
    print("=" * 70)

    try:

        # ====================================================
        # 1. GEMINI CHECK
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
        # 2. READ IMAGE
        # ====================================================

        image_bytes = await file.read()

        print(
            "Received filename:",
            file.filename
        )

        print(
            "Received MIME type:",
            file.content_type
        )

        print(
            "Image size:",
            len(image_bytes),
            "bytes"
        )

        # ====================================================
        # 3. EMPTY IMAGE
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
        # 4. SIZE CHECK
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
        # 5. MIME HANDLING
        # ====================================================
        #
        # Flutter explicitly sends image/jpeg.
        #
        # However, Android/device implementations can
        # occasionally report:
        #
        # application/octet-stream
        # empty MIME
        # image/jpg
        #
        # We therefore do NOT reject the image merely
        # because its MIME metadata is unusual.
        #
        # ====================================================

        received_mime = (
            file.content_type or ""
        ).lower().strip()

        if received_mime in SUPPORTED_MIME_TYPES:

            mime_type = SUPPORTED_MIME_TYPES[
                received_mime
            ]

        else:

            print(
                "Unrecognized MIME type:",
                repr(received_mime)
            )

            print(
                "Falling back to image/jpeg"
            )

            mime_type = "image/jpeg"

        print(
            "MIME type sent to Gemini:",
            mime_type
        )

        # ====================================================
        # 6. GEMINI
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
        # 7. NO ACCEPTED PREDICTION
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

                "prediction":
                    "",

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "status":
                    status

            }

        # ====================================================
        # 8. FINAL SAFETY CHECK
        # ====================================================

        if word not in SUPPORTED_CLASSES_SET:

            print(
                "FINAL SAFETY CHECK FAILED:",
                word
            )

            return {

                "prediction":
                    "",

                "confidence":
                    0.0,

                "status":
                    "Invalid classification"

            }

        # ====================================================
        # 9. SUCCESS
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
                    "",

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