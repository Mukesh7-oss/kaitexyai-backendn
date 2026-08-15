from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp

import os
import asyncio

from pydantic import BaseModel
from openai import OpenAI


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "model/sign_model.pt"

LABELS = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

INPUT_SIZE = 63


# =========================================================
# OPENAI
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini"
)

if OPENAI_API_KEY:

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    print("OpenAI client initialized")

else:

    client = None

    print(
        "WARNING: OPENAI_API_KEY not found."
    )


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Kaitexy AI Sign Language Backend",
    version="3.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# STATIC FILES
# =========================================================

STATIC_FOLDER = "static"

os.makedirs(
    STATIC_FOLDER,
    exist_ok=True
)

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_FOLDER
    ),
    name="static"
)


# =========================================================
# PYTORCH MODEL
# =========================================================

class SignModel(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes
    ):

        super(SignModel, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(
                input_size,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        return self.model(x)


# =========================================================
# LOAD MODEL
# =========================================================

model = SignModel(
    INPUT_SIZE,
    len(LABELS)
)

MODEL_READY = False


try:

    if os.path.exists(MODEL_PATH):

        state_dict = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )

        model.load_state_dict(
            state_dict
        )

        model.eval()

        MODEL_READY = True

        print(
            "Alphabet PyTorch model loaded successfully."
        )

    else:

        print(
            "MODEL NOT FOUND:",
            MODEL_PATH
        )


except Exception as e:

    print(
        "MODEL LOADING ERROR:",
        e
    )


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    model_complexity=0,

    min_detection_confidence=0.3,

    min_tracking_confidence=0.3
)


# =========================================================
# LANDMARK EXTRACTION
# =========================================================

def extract_landmarks(image_bytes):

    try:

        # -------------------------------------------------
        # Convert uploaded image bytes to NumPy
        # -------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        img = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return None


        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        img = cv2.resize(
            img,
            (160, 120)
        )


        # -------------------------------------------------
        # BGR -> RGB
        # -------------------------------------------------

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        # -------------------------------------------------
        # MediaPipe
        # -------------------------------------------------

        results = hands.process(
            img_rgb
        )


        if not results.multi_hand_landmarks:

            return None


        # -------------------------------------------------
        # Extract 21 landmarks
        # 21 x 3 = 63 features
        # -------------------------------------------------

        landmarks = []

        for lm in (
            results
            .multi_hand_landmarks[0]
            .landmark
        ):

            landmarks.extend([

                lm.x,
                lm.y,
                lm.z
            ])


        if len(landmarks) != 63:

            return None


        return np.array(
            landmarks,
            dtype=np.float32
        )


    except Exception as e:

        print(
            "Landmark extraction error:",
            e
        )

        return None


# =========================================================
# PYTORCH PREDICTION
# =========================================================

def predict_landmarks(landmarks):

    if not MODEL_READY:

        return None, 0.0


    try:

        # -------------------------------------------------
        # Shape:
        # (63,) -> (1,63)
        # -------------------------------------------------

        data = np.expand_dims(
            landmarks,
            axis=0
        )


        tensor = torch.from_numpy(
            data
        )


        # -------------------------------------------------
        # Model inference
        # -------------------------------------------------

        with torch.no_grad():

            output = model(
                tensor
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )


            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        index = prediction.item()

        score = confidence.item()


        if (
            index < 0
            or index >= len(LABELS)
        ):

            return None, 0.0


        letter = LABELS[index]


        return letter, score


    except Exception as e:

        print(
            "PyTorch prediction error:",
            e
        )

        return None, 0.0


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {

        "message":
            "Kaitexy AI Backend Running",

        "model_ready":
            MODEL_READY,

        "mode":
            "continuous alphabet recognition",

        "letters":
            len(LABELS),

        "gpt_enabled":
            client is not None,

        "prediction_endpoint":
            "/predict-sign",

        "correction_endpoint":
            "/correct-text"
    }


# =========================================================
# CONTINUOUS SIGN PREDICTION
# =========================================================

@app.post(
    "/predict-sign"
)
async def predict_sign(

    file: UploadFile = File(...)
):

    try:

        # -------------------------------------------------
        # Check model
        # -------------------------------------------------

        if not MODEL_READY:

            return JSONResponse(

                {

                    "prediction": "",

                    "confidence": 0.0,

                    "status":
                        "Model not ready"

                },

                status_code=503
            )


        # -------------------------------------------------
        # Read camera frame
        # -------------------------------------------------

        image_bytes = await file.read()


        if not image_bytes:

            return {

                "prediction": "",

                "confidence": 0.0,

                "status":
                    "Empty image"

            }


        # -------------------------------------------------
        # MediaPipe landmark extraction
        # -------------------------------------------------

        landmarks = extract_landmarks(
            image_bytes
        )


        if landmarks is None:

            return {

                "prediction": "",

                "confidence": 0.0,

                "status":
                    "No hand detected"

            }


        # -------------------------------------------------
        # PyTorch prediction
        # -------------------------------------------------

        letter, confidence = (
            predict_landmarks(
                landmarks
            )
        )


        if letter is None:

            return {

                "prediction": "",

                "confidence": 0.0,

                "status":
                    "Prediction failed"

            }


        # -------------------------------------------------
        # Return prediction
        #
        # IMPORTANT:
        # Frontend performs stabilization.
        # Backend does NOT perform stabilization.
        # -------------------------------------------------

        return {

            "prediction":
                letter,

            "confidence":
                confidence,

            "status":
                "Prediction successful"

        }


    except Exception as e:

        print(
            "Predict endpoint error:",
            e
        )


        return JSONResponse(

            {

                "prediction": "",

                "confidence": 0.0,

                "status":
                    "Server error",

                "error":
                    str(e)

            },

            status_code=500
        )


# =========================================================
# GPT WORD CORRECTION
# =========================================================

class CorrectTextRequest(BaseModel):

    text: str


@app.post(
    "/correct-text"
)
async def correct_text(
    request: CorrectTextRequest
):

    raw_text = request.text.strip()


    # -----------------------------------------------------
    # Empty text
    # -----------------------------------------------------

    if not raw_text:

        return {

            "corrected_text": "",

            "status":
                "Empty text"

        }


    # -----------------------------------------------------
    # GPT unavailable
    # -----------------------------------------------------

    if client is None:

        return {

            "corrected_text":
                raw_text,

            "status":
                "GPT unavailable"

        }


    try:

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt = f"""
You are a spelling correction component
inside a sign-language alphabet recognition
application.

The vision model produced this word:

{raw_text}

Correct obvious recognition and spelling mistakes.

Rules:

1. Preserve the intended word.
2. Correct obvious recognition mistakes.
3. Correct spelling.
4. Do not invent unrelated words.
5. Do not add explanations.
6. Do not add punctuation.
7. Return ONLY the corrected word.
8. Use normal English capitalization.
"""


        # -------------------------------------------------
        # GPT
        # -------------------------------------------------

        response = await asyncio.to_thread(

            lambda:
            client.responses.create(

                model=OPENAI_MODEL,

                input=prompt,

                max_output_tokens=20
            )
        )


        corrected = (
            response
            .output_text
            .strip()
        )


        if not corrected:

            corrected = raw_text


        # -------------------------------------------------
        # Return
        # -------------------------------------------------

        return {

            "corrected_text":
                corrected,

            "status":
                "Word corrected"

        }


    except Exception as e:

        print(
            "GPT correction error:",
            e
        )


        # -------------------------------------------------
        # IMPORTANT:
        # If GPT fails, frontend still gets
        # the original word.
        # -------------------------------------------------

        return {

            "corrected_text":
                raw_text,

            "status":
                "GPT correction failed",

            "error":
                str(e)

        }


# =========================================================
# TEXT TO SIGN
# =========================================================

@app.get(
    "/text-to-sign"
)
def text_to_sign(
    text: str
):

    clean_text = (

        text
        .lower()
        .strip()
        .replace(" ", "")
    )


    image_path = (

        f"static/signs/"
        f"{clean_text}.png"
    )


    if not os.path.exists(
        image_path
    ):

        return JSONResponse(

            {

                "error":
                    f"Sign for '{text}' not found"

            },

            status_code=404
        )


    return {

        "image_url":
            f"/static/signs/"
            f"{clean_text}.png"

    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get(
    "/health"
)
def health():

    return {

        "status":
            "healthy",

        "model_ready":
            MODEL_READY,

        "gpt_enabled":
            client is not None

    }


# =========================================================
# RUN
# =========================================================

# Local:
#
# uvicorn main:app --host 0.0.0.0 --port 10000
#
# Render:
#
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# =========================================================