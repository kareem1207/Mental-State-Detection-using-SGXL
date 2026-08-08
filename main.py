from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uuid
import os

from utils.speech_to_text import transcribe_audio
from utils.language_detect import detect_language
from utils.translator import translate_text
from utils.tts import generate_audio
from ml_pipeline import analyze_mental_state

# ----------------- APP SETUP -----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        # 4-7. Encode, classify, and look up the canned response
        result = analyze_mental_state(english_text)
        prediction = result["prediction"]

        # 8. Translate audio text back to user language
        localized_audio_text = translate_text(
            result["audio_text"],
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
            "tips": result.get("tips", []),
            "yoga": result.get("yoga", []),
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

