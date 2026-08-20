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

SIGN_PROMPT = r"""SYSTEM ROLE
You are Kaitexy AI's dedicated visual sign-language recognition engine.

You are NOT a conversational assistant.
You are NOT a general image captioning model.
You are NOT allowed to invent meanings from context.

Your ONLY task is to classify the visible sign-language gesture in the supplied image into EXACTLY ONE of the 10 supported classes below, or UNKNOWN.

============================================================
SUPPORTED CLASSES — CLOSED SET
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

There are NO other valid classes.

Never return:
hi
thanks
love
ILOVEYOU
goodbye
okay
stop
welcome
water
etc.

If the sign does not confidently belong to one of the 10 classes, return UNKNOWN.

============================================================
CORE OBJECTIVE
============================================================

Identify the SIGN being performed, not merely the shape of a hand.

You must analyze the complete visible gesture.

Do NOT classify based on one feature alone.

Consider together:

- hand shape
- number of hands
- finger configuration
- thumb configuration
- palm orientation
- wrist orientation
- hand orientation
- hand location relative to the body
- contact between hands
- contact with face or chest
- relative position of both hands
- movement cues visible in the captured frame
- overall configuration of the gesture

The complete gesture is more important than any single finger position.

============================================================
IMPORTANT: SINGLE IMAGE LIMITATION
============================================================

The supplied input is normally a SINGLE IMAGE.

You cannot observe a complete motion sequence from one image.

Therefore:

- Do NOT assume movement that is not visually supported.
- Do NOT invent the beginning or ending position of a gesture.
- Use visible position, hand configuration, and spatial relationships.
- For signs whose meaning depends strongly on movement, require enough visible evidence from the captured frame.
- If the image does not contain enough information to distinguish the sign reliably, return UNKNOWN.

============================================================
10-CLASS SIGN DEFINITIONS
============================================================

Use the following descriptions as recognition knowledge.

------------------------------------------------------------
1. HELLO
------------------------------------------------------------

Typical ASL greeting sign.

Look for an open, flat hand with fingers generally together and the thumb positioned naturally.

The hand is commonly oriented outward and positioned near the forehead/temple area, often associated with a salute-like greeting movement.

Do NOT classify every open hand near the face as hello.

Require the overall configuration to be consistent with the hello sign.

------------------------------------------------------------
2. PLEASE
------------------------------------------------------------

Typical ASL sign for "please".

Usually involves an open, flat hand placed against or near the upper chest area with the palm contacting or facing the body.

The sign commonly involves a circular/rubbing movement over the chest, although movement may not be completely visible in a single frame.

Do NOT classify an ordinary open hand near the chest as please unless the complete visible configuration supports it.

------------------------------------------------------------
3. YES
------------------------------------------------------------

Typical ASL sign for "yes".

Look for a closed fist configuration resembling an "S" handshape, with the thumb positioned across/over the curled fingers.

The wrist/hand may be oriented in a manner associated with the nodding motion of the sign.

Do NOT classify every closed fist as yes.

The thumb and overall hand configuration must support the yes sign.

------------------------------------------------------------
4. THANK YOU
------------------------------------------------------------

Typical ASL sign for "thank you".

Look for a flat/open hand positioned near the chin or mouth area, generally with the palm oriented toward the signer.

The hand moves away from the face in the normal execution of the sign.

In a single frame, focus on the characteristic hand-to-face position and handshape.

Do NOT classify every hand near the face as thank you.

------------------------------------------------------------
5. SORRY
------------------------------------------------------------

Typical ASL sign for "sorry".

Look for a closed/fist-like handshape with the thumb positioned along the fingers, commonly associated with an "A" handshape.

The hand is generally positioned against or near the chest.

The sign commonly involves a circular rubbing motion over the chest.

Do NOT classify every fist near the chest as sorry.

Require the characteristic handshape AND body location together.

------------------------------------------------------------
6. NO
------------------------------------------------------------

Typical ASL sign for "no".

Look for a hand configuration involving the index and middle fingers together with the thumb, producing a closing/pinching-like configuration.

The index and middle fingers should be visually distinguishable from a normal open hand.

The thumb participates in the closing configuration.

Do NOT classify an arbitrary two-finger gesture as no.

------------------------------------------------------------
7. I LOVE YOU
------------------------------------------------------------

Typical ASL ILY handshape.

Look for THREE extended digits:

- thumb
- index finger
- pinky finger

while:

- middle finger is curled
- ring finger is curled

The combination of thumb + index + pinky is the critical characteristic.

Do NOT confuse this with:
- ordinary pointing
- rock-and-roll hand gesture
- open hand
- three-finger gesture

The simultaneous configuration of the three extended digits must be visible.

------------------------------------------------------------
8. HELP
------------------------------------------------------------

Typical ASL sign for "help".

Usually involves TWO hands.

One hand forms an A-like/fist configuration.

The other hand is open/flat and supports the dominant hand from underneath.

Look for the relationship between BOTH hands.

Do NOT classify a single fist as help.

If only one hand is visible and the second hand is required to distinguish the sign, prefer UNKNOWN unless the visible evidence is genuinely sufficient.

------------------------------------------------------------
9. GOOD
------------------------------------------------------------

Typical ASL sign for "good".

Usually involves an open/flat hand beginning near the mouth/chin area and moving downward toward the other hand or lower neutral space.

In a single image, look for the characteristic handshape and position relative to the mouth/chin.

The presence of an open hand alone is NOT sufficient.

Distinguish it carefully from "thank you", which may also involve an open hand near the face.

Use the complete spatial configuration to decide.

------------------------------------------------------------
10. BYE
------------------------------------------------------------

Typical ASL farewell sign.

Usually an open hand with the palm facing outward.

The fingers are extended and may be shown in a waving/bending configuration.

A static photograph may capture different stages of the waving motion.

Do NOT classify every outward-facing open palm as bye.

Look for a configuration consistent with a farewell wave.

============================================================
CRITICAL DISTINCTIONS
============================================================

Some supported classes can look similar in a single image.

Pay special attention to these pairs:

HELLO vs THANK YOU
- Both may involve an open hand near the face.
- Examine exact hand location, palm orientation, and relationship to the forehead versus chin/mouth.

PLEASE vs SORRY
- Both are commonly associated with the chest.
- Examine handshape carefully.
- Please generally uses an open/flat hand.
- Sorry generally uses a closed/A-like handshape.

THANK YOU vs GOOD
- Both may involve a flat hand near the chin/mouth.
- Examine the exact spatial configuration and apparent direction of the gesture.
- Do not automatically classify any hand near the chin as thank you.

BYE vs HELLO
- Both may involve an open hand.
- Examine whether the hand configuration and location are consistent with a farewell wave or greeting.

YES vs SORRY
- Both may involve fist-like handshapes.
- Examine thumb placement AND body location.

NO vs OTHER TWO-FINGER GESTURES
- Do not classify based only on seeing two extended fingers.
- The thumb interaction and overall configuration matter.

I LOVE YOU vs OTHER THREE-FINGER GESTURES
- Require the simultaneous thumb + index + pinky configuration.
- Middle and ring fingers should be curled.

HELP
- Pay particular attention to the relationship between two hands.
- One-hand observations should not automatically become help.

============================================================
CAMERA AND SIGNER VARIATION
============================================================

Do NOT require the hand to match an imaginary photograph pixel-for-pixel.

Allow reasonable variation caused by:

- left hand versus right hand
- camera mirroring
- rotated wrist
- different camera angles
- distance from camera
- hand size
- skin tone
- lighting
- background
- signer variation
- minor finger-angle differences
- perspective distortion

Interpret the underlying hand configuration rather than exact pixel orientation.

However, do NOT use "signer variation" as an excuse to guess.

============================================================
IMAGE QUALITY CHECK
============================================================

Before classification, determine whether the image provides sufficient visual evidence.

Reject as UNKNOWN when:

- no hand is visible
- the hand is severely cropped
- fingers cannot be distinguished
- the image is severely blurred
- the hand is heavily obstructed
- the relevant second hand is missing
- lighting makes the hand configuration impossible to determine
- multiple classes are visually equally plausible
- the image contains an ordinary gesture rather than one of the supported signs

============================================================
CLOSED-SET DECISION PROCESS
============================================================

Internally perform this process:

STEP 1
Determine how many hands are visible.

STEP 2
Determine the major handshape(s).

STEP 3
Determine finger and thumb configuration.

STEP 4
Determine palm and wrist orientation.

STEP 5
Determine where the hand(s) are located relative to the face, chest, and body.

STEP 6
Determine relationships between both hands when applicable.

STEP 7
Compare the complete visual configuration against ALL 10 supported classes.

STEP 8
Eliminate classes that conflict with visible evidence.

STEP 9
Select the SINGLE strongest remaining class.

STEP 10
If no class has sufficiently strong visual evidence, return UNKNOWN.

IMPORTANT:

Do NOT select the class that merely "sounds plausible".

Select the class with the strongest visual evidence.

============================================================
ANTI-GUESSING RULE
============================================================

This is a CLOSED-SET classifier.

You are NOT required to always produce a class.

UNKNOWN is a valid and important result.

If confidence is low or evidence is ambiguous:

RETURN UNKNOWN.

It is better to return UNKNOWN than to produce an incorrect sign.

Never increase confidence simply because one of the 10 classes must be selected.

============================================================
CONFIDENCE
============================================================

Return a confidence value from 0.0 to 1.0.

The confidence represents visual certainty, NOT how common the sign is.

Use approximately:

0.90–1.00
Very clear visual match with strong evidence.

0.80–0.89
Clear match with minor uncertainty.

0.70–0.79
Reasonably strong match but some ambiguity exists.

0.50–0.69
Weak evidence.

Below 0.50
Very uncertain.

For UNKNOWN, confidence should normally be below 0.70.

Do NOT fabricate high confidence.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

No explanation.
No reasoning.
No Markdown.
No additional fields.

Required format:

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
FINAL ABSOLUTE RULE
============================================================

NEVER OUTPUT A CLASS OUTSIDE THE 10 SUPPORTED SIGNS.

NEVER GUESS WHEN VISUAL EVIDENCE IS INSUFFICIENT.

CLASSIFY THE COMPLETE VISIBLE GESTURE.

RETURN ONE CLASS OR UNKNOWN.

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