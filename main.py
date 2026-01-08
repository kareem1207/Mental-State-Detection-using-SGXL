from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uuid, json, joblib
import os

from utils.speech_to_text import transcribe_audio
from utils.language_detect import detect_language
from utils.translator import translate_text
from utils.tts import generate_audio

# ----------------- APP SETUP -----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- LOAD MODEL PIPELINE -----------------
pipeline = joblib.load("model/final_pipeline.joblib")

embedder = pipeline["embedder"]
clf_model = pipeline["model"]
# Convert to list to avoid numpy indexing issues
classes = list(pipeline["classes"]) if hasattr(pipeline["classes"], '__iter__') else pipeline["classes"]

# ----------------- LOAD JSON SESSIONS -----------------
with open("data/sessions.json", "r", encoding="utf-8") as f:
    sessions = json.load(f)

# ----------------- SERVE STATIC FILES -----------------
# Mount audio output folder
if os.path.exists("audio/output"):
    app.mount("/audio", StaticFiles(directory="audio/output"), name="audio")

# Mount audio input folder for user audio playback
if os.path.exists("audio/input"):
    app.mount("/user-audio", StaticFiles(directory="audio/input"), name="user-audio")

# ----------------- ROOT ENDPOINT -----------------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>Mental Health Audio Analyzer</h1>
                <p>API is running. Frontend not found. Please create static/index.html</p>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Mental Health Analyzer API is running"}

# ----------------- API ENDPOINT -----------------
@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    try:
        # Ensure directories exist
        os.makedirs("audio/input", exist_ok=True)
        os.makedirs("audio/output", exist_ok=True)
        
        # Save uploaded audio
        input_audio_path = f"audio/input/{uuid.uuid4()}.wav"
        with open(input_audio_path, "wb") as f:
            f.write(await audio.read())

        # 1. Speech → Text
        original_text = transcribe_audio(input_audio_path)

        # 2. Detect language
        user_lang = detect_language(original_text)

        # 3. Translate to English
        english_text = translate_text(original_text, src=user_lang, tgt="en")

        # 4. Encode text (IMPORTANT FIX)
        X_emb = embedder.encode(
            [english_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # 5. Predict class index
        pred_result = clf_model.predict(X_emb)
        pred_idx = pred_result[0]
        
        # Convert to Python int to avoid numpy indexing issues
        try:
            pred_idx = int(pred_idx)
        except (TypeError, ValueError):
            pred_idx = 0  # Default to first class if conversion fails

        # 6. Map index → class label
        if isinstance(classes, (list, tuple)):
            prediction = classes[pred_idx]
        else:
            # Handle numpy array or other array-like objects
            prediction = str(classes[pred_idx])

        # 7. Load response data
        session = sessions[prediction]

        # 8. Translate audio text back to user language
        localized_audio_text = translate_text(
            session["audio_text"],
            src="en",
            tgt=user_lang
        )

        # 9. Generate audio
        audio_output_path = generate_audio(localized_audio_text)
        
        # Get filenames for URLs
        audio_filename = os.path.basename(audio_output_path)
        input_audio_filename = os.path.basename(input_audio_path)

        return {
            "prediction": prediction,
            "transcribed_text": original_text,
            "english_text": english_text,
            "response_text": localized_audio_text,
            "tips": session.get("tips", []),
            "yoga": session.get("yoga", []),
            "audio_url": f"/audio/{audio_filename}",
            "user_audio_url": f"/user-audio/{input_audio_filename}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "message": "An error occurred during analysis. Please try again.",
            "prediction": "Error",
            "transcribed_text": "",
            "response_text": str(e),
            "tips": [],
            "yoga": [],
            "audio_url": "",
            "user_audio_url": ""
        }

