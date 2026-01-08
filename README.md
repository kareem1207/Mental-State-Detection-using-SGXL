# 🧠 Mental Health Audio Analyzer

An AI-powered mental health analysis tool that processes audio input, transcribes speech, detects language, analyzes mental state, and provides personalized guidance.

## ✨ Features

- 🎤 **Audio Recording & Upload**: Record directly or upload audio files
- 🗣️ **Speech-to-Text**: Automatic transcription using OpenAI Whisper
- 🌐 **Multi-language Support**: Automatic language detection and translation
- 🤖 **ML Classification**: Analyzes mental state (Normal, Depression, Anxiety, Suicidal)
- 💡 **Personalized Guidance**: Provides tips and yoga recommendations
- 🔊 **Audio Response**: Text-to-speech feedback in user's language
- 🎨 **Modern UI**: Beautiful, responsive web interface

## 🚀 Setup

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1.**Clone the repository**

```bash
git clone https://github.com/kareem1207/Mental-State-Detection-using-SGXL.git
cd Mental-State-Detection-using-SGXL
```

2.**Activate virtual environment**

3.**Install dependencies**

```bash
pip install -r requirements.txt
```

if you have uv installed use :

```bash
uv pip install -r requirements.txt
```

1.**Ensure model files exist**
- The Whisper model should be at: `model/openai whisper/small.pt`
- The ML pipeline should be at: `model/final_pipeline.joblib`

## 🎯 Running the Application

### Start the server

```bash
uvicorn main:app --reload
```

The server will start at: `http://127.0.0.1:8000`

### Access the application

- **Web Interface**: <http://127.0.0.1:8000>
- **API Documentation**: <http://127.0.0.1:8000/docs>
- **Health Check**: <http://127.0.0.1:8000/health>

## 📁 Project Structure

```md
Mental-State-Detection-using-SGXL/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── static/
│   └── index.html         # Frontend interface
├── utils/
│   ├── speech_to_text.py  # Whisper transcription
│   ├── language_detect.py # Language detection
│   ├── translator.py      # Translation service
│   └── tts.py            # Text-to-speech
├── model/
│   ├── final_pipeline.joblib
│   └── openai whisper/
│       └── small.pt
├── data/
│   └── sessions.json     # Response templates
└── audio/
    ├── input/            # Uploaded audio
    └── output/           # Generated audio
```

## 🔧 API Endpoints

### POST /analyze

Analyze audio and get mental health assessment.

**Request**: Multipart form data with audio file

**Response**:

```json
{
  "prediction": "Normal",
  "transcribed_text": "Original user speech",
  "english_text": "Translated to English",
  "response_text": "Personalized response in user's language",
  "tips": ["Tip 1", "Tip 2"],
  "yoga": ["Yoga 1", "Yoga 2"],
  "audio_url": "/audio/response.wav"
}
```

### GET /health

Check if the API is running.

## 🛠️ Technologies Used

- **Backend**: FastAPI, Uvicorn
- **ML/AI**:
  - OpenAI Whisper (speech-to-text)
  - Sentence Transformers (embeddings)
  - Scikit-learn (classification)
  - Transformers (translation)
- **Audio**: pyttsx3 (text-to-speech)
- **Frontend**: HTML, CSS, JavaScript

## 📊 Mental Health Categories

1. **Normal**: Healthy mental state
2. **Depression**: Signs of depression detected
3. **Anxiety**: Anxiety indicators present
4. **Suicidal**: Critical state requiring immediate attention

## Outputs

 [!Output 1](./output/output%201.png)

 [!Output 2](./output/output%202.png)
 [!Output 3](./output/output%203.png)

## ⚠️ Important Notes

- This is an educational/research tool, not a replacement for professional mental health services
- If in crisis, please contact a mental health professional immediately
- The model predictions should be used as guidance only

## 🐛 Troubleshooting

### Whisper Model Error

- Ensure model path is correct: `model/openai whisper/small.pt`
- Download model if missing using Whisper CLI

### Audio Not Playing

- Check that audio files are being generated in `audio/output/`
- Verify pyttsx3 is properly installed

### Translation Errors

- Ensure `transformers` and dependencies are installed
- Check internet connection for first-time model download

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---
**Note**: Always seek professional help for mental health concerns. This tool is for educational purposes only.
