import pyttsx3
import uuid

engine = pyttsx3.init()

def generate_audio(text: str) -> str:
    output_path = f"audio/output/{uuid.uuid4()}.wav"
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path
