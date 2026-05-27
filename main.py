from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.inference.speaker import SpeakerRecognition
import shutil
import os
import uuid
import numpy as np
from scipy.spatial.distance import cosine

app = FastAPI()

# Permitir conexiones desde tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

# Cargar modelo biométrico
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Carpeta usuarios
USERS_DIR = "users"

if not os.path.exists(USERS_DIR):
    os.makedirs(USERS_DIR)

# =====================================================
# FUNCION PARA EXTRAER EMBEDDING
# =====================================================

def get_embedding(audio_path):
    embedding = verification.encode_file(audio_path)
    return embedding.detach().numpy()

# =====================================================
# REGISTRAR VOZ
# =====================================================

@app.post("/enroll")
async def enroll(
    username: str = Form(...),
    audio: UploadFile = File(...)
):

    temp_filename = f"temp_{uuid.uuid4()}.wav"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:

        embedding = get_embedding(temp_filename)

        user_folder = os.path.join(USERS_DIR, username)

        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        np.save(
            os.path.join(user_folder, "embedding.npy"),
            embedding
        )

        os.remove(temp_filename)

        return {
            "success": True,
            "message": f"Voz registrada para {username}"
        }

    except Exception as e:

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        return {
            "success": False,
            "error": str(e)
        }

# =====================================================
# VERIFICAR VOZ
# =====================================================

@app.post("/verify")
async def verify(
    audio: UploadFile = File(...)
):

    temp_filename = f"temp_{uuid.uuid4()}.wav"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:

        new_embedding = get_embedding(temp_filename)

        best_score = 0
        best_user = None

        for username in os.listdir(USERS_DIR):

            embedding_path = os.path.join(
                USERS_DIR,
                username,
                "embedding.npy"
            )

            if not os.path.exists(embedding_path):
                continue

            saved_embedding = np.load(embedding_path)

            similarity = 1 - cosine(
                saved_embedding.flatten(),
                new_embedding.flatten()
            )

            if similarity > best_score:
                best_score = similarity
                best_user = username

        os.remove(temp_filename)

        THRESHOLD = 0.65

        if best_score >= THRESHOLD:

            return {
                "match": True,
                "user": best_user,
                "score": float(best_score)
            }

        return {
            "match": False,
            "user": None,
            "score": float(best_score)
        }

    except Exception as e:

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        return {
            "success": False,
            "error": str(e)
        }

# =====================================================
# ROOT
# =====================================================

@app.get("/")
async def root():
    return {
        "status": "Speak2Speak Biometric API Running"
    }