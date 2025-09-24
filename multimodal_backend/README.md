# Multimodal Processing API (FastAPI)

A demonstration backend that accepts and processes multimodal inputs (text, image, audio) with clean, organized FastAPI patterns and Ocean Professional themed docs.

Endpoints
- GET /             Health check
- GET /info         Service info and theme metadata
- POST /text/analyze
- POST /image/process (multipart/form-data)
- POST /audio/process (multipart/form-data)
- GET /realtime/docs WebSocket usage notes (HTML)

Run locally
1. Create a virtual environment and install requirements:
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt

2. Start server:
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 3001

Environment variables (optional)
- CORS_ALLOW_ORIGINS: Comma-separated origins (default "*")

cURL examples
- Text:
  curl -X POST http://localhost:3001/text/analyze \
    -H "Content-Type: application/json" \
    -d '{"text":"I love FastAPI, it is amazing and great!","language":"en","sentiment":true,"keywords":true}'

- Image:
  curl -X POST http://localhost:3001/image/process \
    -F "file=@/path/to/image.jpg" \
    -F "hint=dog in a park"

- Audio:
  curl -X POST http://localhost:3001/audio/process \
    -F "file=@/path/to/audio.mp3" \
    -F "prompt=casual phone call transcription"

Notes
- This app performs mock processing for demonstration.
- OpenAPI docs are available at /docs
