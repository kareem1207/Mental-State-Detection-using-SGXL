import whisper
import os

MODEL_PATH = r"model/openai whisper/small.pt"

model = whisper.load_model(MODEL_PATH)

def transcribe_audio(audio_path: str) -> str:
    result = model.transcribe(audio_path)
    return str(result["text"])
