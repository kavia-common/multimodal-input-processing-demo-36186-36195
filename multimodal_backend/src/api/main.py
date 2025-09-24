import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, constr, confloat

# PUBLIC_INTERFACE
def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application with routes and middleware.
    """
    app = FastAPI(
        title="Multimodal Processing API",
        description=(
            "A demo API that accepts and processes multimodal inputs (text, image, audio). "
            "This backend demonstrates organized, well-documented FastAPI patterns with Ocean Professional styling."
        ),
        version="1.0.0",
        contact={"name": "Multimodal Demo", "url": "https://example.com"},
        license_info={"name": "MIT"},
        terms_of_service="https://example.com/terms",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 0,
            "docExpansion": "list",
            "filter": True,
            "syntaxHighlight.theme": "monokai",
        },
        swagger_ui_oauth2_redirect_url=None,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenAPI tags for grouping
    openapi_tags = [
        {"name": "Health", "description": "Service status and metadata endpoints."},
        {"name": "Text", "description": "Textual analysis and utilities."},
        {"name": "Image", "description": "Image processing endpoints."},
        {"name": "Audio", "description": "Audio processing endpoints."},
        {"name": "Realtime", "description": "WebSocket usage and notes."},
    ]
    app.openapi_tags = openapi_tags

    # Models
    class HealthResponse(BaseModel):
        status: str = Field(..., description="Service status string.")
        timestamp: str = Field(..., description="ISO timestamp of the server response.")
        version: str = Field(..., description="API version.")

    class InfoResponse(BaseModel):
        name: str = Field(..., description="Service name.")
        theme: str = Field(..., description="Active style theme.")
        colors: dict = Field(..., description="Theme color palette.")
        features: List[str] = Field(..., description="Supported feature list.")
        docs: str = Field(..., description="Docs URL.")
        openapi: str = Field(..., description="OpenAPI JSON URL.")

    class TextAnalyzeRequest(BaseModel):
        text: constr(min_length=1) = Field(..., description="Input text to analyze.")
        language: Optional[constr(strip_whitespace=True, min_length=2, max_length=5)] = Field(
            default="en", description="Language code for hints (e.g., en, es)."
        )
        sentiment: bool = Field(
            default=True, description="Whether to include mock sentiment analysis."
        )
        keywords: bool = Field(
            default=True, description="Whether to include mock keyword extraction."
        )

    class TextAnalyzeResponse(BaseModel):
        length: int = Field(..., description="Length of input text in characters.")
        word_count: int = Field(..., description="Number of whitespace-delimited tokens.")
        language: str = Field(..., description="Language hint used.")
        sentiment: Optional[dict] = Field(None, description="Mock sentiment analysis result.")
        keywords: Optional[List[str]] = Field(None, description="Mock extracted keywords.")
        summary: str = Field(..., description="Mock summary of the text.")

    class ImageProcessResponse(BaseModel):
        filename: str = Field(..., description="Original filename.")
        content_type: str = Field(..., description="MIME type of the uploaded file.")
        size_bytes: int = Field(..., description="File size in bytes.")
        detected_objects: List[str] = Field(..., description="List of mock detected objects.")
        dominant_colors: List[str] = Field(..., description="Mock dominant colors.")
        message: str = Field(..., description="Processing summary message.")

    class AudioProcessResponse(BaseModel):
        filename: str = Field(..., description="Original filename.")
        content_type: str = Field(..., description="MIME type of the uploaded file.")
        size_bytes: int = Field(..., description="File size in bytes.")
        duration_seconds: confloat(ge=0) = Field(..., description="Mock estimated duration.")
        transcript_preview: str = Field(..., description="Mock transcript preview.")
        message: str = Field(..., description="Processing summary message.")

    # Routes
    @app.get("/", response_model=HealthResponse, tags=["Health"], summary="Health Check", description="Returns basic health status for the service.")
    # PUBLIC_INTERFACE
    def health_check():
        """Health check endpoint.
        Returns:
            HealthResponse: status, timestamp, and version.
        """
        return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat() + "Z", version=app.version)

    @app.get(
        "/info",
        response_model=InfoResponse,
        tags=["Health"],
        summary="Service Info",
        description="Provides service name, theme, palette, supported features, and docs links.",
    )
    # PUBLIC_INTERFACE
    def service_info():
        """Service information endpoint for metadata and theme."""
        theme = {
            "name": "Ocean Professional",
            "primary": "#2563EB",
            "secondary": "#F59E0B",
            "success": "#F59E0B",
            "error": "#EF4444",
            "background": "#f9fafb",
            "surface": "#ffffff",
            "text": "#111827",
        }
        base = ""
        return InfoResponse(
            name="Multimodal Processing API",
            theme=theme["name"],
            colors=theme,
            features=["text.analyze", "image.process", "audio.process"],
            docs=f"{base}/docs",
            openapi=f"{base}/openapi.json",
        )

    @app.post(
        "/text/analyze",
        response_model=TextAnalyzeResponse,
        tags=["Text"],
        summary="Analyze text",
        description="Accepts raw text and returns mock analysis: token counts, sentiment, keywords, and a brief summary.",
        responses={
            200: {"description": "Successful analysis", "model": TextAnalyzeResponse},
            400: {"description": "Invalid input"},
        },
    )
    # PUBLIC_INTERFACE
    def analyze_text(payload: TextAnalyzeRequest = Body(...)):
        """
        Perform a mock analysis on provided text.
        Args:
            payload: TextAnalyzeRequest containing text and options.
        Returns:
            TextAnalyzeResponse with mock results.
        """
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text must not be empty.")

        tokens = text.split()
        word_count = len(tokens)

        # Mock sentiment: based on simple heuristics
        positive_words = {"good", "great", "excellent", "amazing", "love", "happy"}
        negative_words = {"bad", "terrible", "awful", "hate", "sad", "angry"}
        pos_hits = sum(1 for t in tokens if t.lower().strip(".,!?") in positive_words)
        neg_hits = sum(1 for t in tokens if t.lower().strip(".,!?") in negative_words)
        score = pos_hits - neg_hits
        sentiment = None
        if payload.sentiment:
            sentiment = {
                "score": score,
                "label": "positive" if score > 0 else ("negative" if score < 0 else "neutral"),
                "positives": pos_hits,
                "negatives": neg_hits,
            }

        # Mock keywords: take top N unique longer tokens
        keywords = None
        if payload.keywords:
            unique_long = sorted({t.strip(".,!?") for t in tokens if len(t.strip(".,!?")) >= 5})
            keywords = unique_long[:5]

        # Mock summary
        if word_count <= 12:
            summary = text
        else:
            summary = " ".join(tokens[:12]) + " …"

        return TextAnalyzeResponse(
            length=len(text),
            word_count=word_count,
            language=payload.language or "en",
            sentiment=sentiment,
            keywords=keywords,
            summary=summary,
        )

    @app.post(
        "/image/process",
        response_model=ImageProcessResponse,
        tags=["Image"],
        summary="Process an image",
        description="Upload an image file. Returns mock detected objects and color info.",
        responses={
            200: {"description": "Processed", "model": ImageProcessResponse},
            400: {"description": "Unsupported media type or invalid file"},
        },
    )
    # PUBLIC_INTERFACE
    async def process_image(
        file: UploadFile = File(..., description="Image file to upload."),
        hint: Optional[str] = Form(None, description="Optional hint describing image content."),
    ):
        """
        Mock image processing that extracts metadata and returns fake detections.
        Args:
            file: Uploaded image file.
            hint: Optional user-provided hint to influence mock detections.
        Returns:
            ImageProcessResponse with mock detected objects and colors.
        """
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a valid image file.")

        # Read bytes only to measure size; avoid storing the entire file elsewhere.
        data = await file.read()
        size = len(data)

        # Very naive mock detections influenced by hint
        base_objects = ["person", "bottle", "chair", "laptop", "plant", "dog"]
        if hint:
            hint_token = hint.lower().strip().split()[0]
            if hint_token not in base_objects:
                base_objects = [hint_token] + base_objects
        detected = base_objects[:3]

        # Mock dominant colors
        colors = ["#2563EB", "#F59E0B", "#34D399"]

        return ImageProcessResponse(
            filename=file.filename or "uploaded_image",
            content_type=file.content_type,
            size_bytes=size,
            detected_objects=detected,
            dominant_colors=colors,
            message="Image processed successfully (mock).",
        )

    @app.post(
        "/audio/process",
        response_model=AudioProcessResponse,
        tags=["Audio"],
        summary="Process an audio file",
        description="Upload an audio file. Returns mock duration and transcript preview.",
        responses={
            200: {"description": "Processed", "model": AudioProcessResponse},
            400: {"description": "Unsupported media type or invalid file"},
        },
    )
    # PUBLIC_INTERFACE
    async def process_audio(
        file: UploadFile = File(..., description="Audio file to upload (wav/mp3/ogg)."),
        prompt: Optional[str] = Form(None, description="Optional prompt to steer transcription."),
    ):
        """
        Mock audio processing that approximates duration and returns a fake transcript preview.
        Args:
            file: Uploaded audio file.
            prompt: Optional prompt to steer mock output.
        Returns:
            AudioProcessResponse including mock duration and transcript preview.
        """
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a valid audio file.")

        data = await file.read()
        size = len(data)

        # Mock duration estimation: size-based heuristic (bytes / bitrate)
        # Assume ~32 kbps -> 4000 bytes/s for mock purposes
        duration_sec = round(size / 4000.0, 2)

        base_transcript = f"Transcribed audio from {file.filename or 'audio'}."
        if prompt:
            transcript_preview = f"{base_transcript} Prompt noted: {prompt[:40]}..."
        else:
            transcript_preview = base_transcript

        return AudioProcessResponse(
            filename=file.filename or "uploaded_audio",
            content_type=file.content_type,
            size_bytes=size,
            duration_seconds=max(duration_sec, 0.01),
            transcript_preview=transcript_preview[:160],
            message="Audio processed successfully (mock).",
        )

    # WebSocket usage doc helper (no live WS in this demo; but document the route style)
    @app.get(
        "/realtime/docs",
        tags=["Realtime"],
        summary="Realtime/WebSocket usage",
        description="Documentation for how a WebSocket endpoint would be used in this project.",
        response_class=HTMLResponse,
    )
    # PUBLIC_INTERFACE
    def websocket_docs():
        """
        Returns HTML content explaining the hypothetical WebSocket usage for real-time multimodal events.
        """
        # Ocean Professional themed minimal HTML
        return HTMLResponse(
            content=f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Realtime Usage</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root {{
      --primary: #2563EB;
      --secondary: #F59E0B;
      --bg: #f9fafb;
      --surface: #ffffff;
      --text: #111827;
      --radius: 12px;
      --shadow: 0 8px 24px rgba(0,0,0,0.08);
    }}
    body {{
      margin: 0; padding: 0; background: var(--bg); color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
    }}
    .container {{
      max-width: 920px; margin: 48px auto; padding: 0 20px;
    }}
    .card {{
      background: var(--surface); border-radius: var(--radius); box-shadow: var(--shadow);
      padding: 28px;
      border: 1px solid #e5e7eb;
    }}
    h1 {{
      margin-top: 0; color: var(--primary);
      background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(243,244,246,0));
      padding: 10px 14px; border-radius: 10px;
    }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}
    pre {{
      background: #0b1021; color: #e5e7eb; padding: 16px; border-radius: 10px; overflow-x: auto;
    }}
    .note {{
      background: #eff6ff; border-left: 4px solid var(--primary); padding: 12px 16px; border-radius: 8px;
    }}
    a.button {{
      display: inline-block; padding: 10px 16px; background: var(--primary); color: white;
      text-decoration: none; border-radius: 10px; box-shadow: var(--shadow);
      transition: transform 0.08s ease, box-shadow 0.2s ease;
    }}
    a.button:hover {{ transform: translateY(-1px); box-shadow: 0 10px 28px rgba(37,99,235,0.25); }}
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>Realtime Usage (WebSocket)</h1>
      <p class="note">
        This demo does not implement a live WebSocket endpoint, but here is how you would typically connect:
      </p>
      <pre><code>// JavaScript example
const ws = new WebSocket("wss://your-host/realtime");
ws.onopen = () =&gt; ws.send(JSON.stringify({ type: "hello", mode: "multimodal" }));
ws.onmessage = (evt) =&gt; console.log("event:", evt.data);
ws.onclose = () =&gt; console.log("closed");</code></pre>
      <p>
        For real implementations, you would broadcast incremental transcription for audio,
        token streams for text, or detection updates for images. Consider tagging messages with
        a <code>type</code> field for routing (e.g., <code>token</code>, <code>end</code>, <code>error</code>).
      </p>
      <a class="button" href="/docs">Open API Docs</a>
    </div>
  </div>
</body>
</html>
            """
        )

    # Custom Ocean Professional Swagger theme (inject CSS)
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        openapi_schema["tags"] = openapi_tags
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore

    # Root redirect to docs for web usage convenience
    @app.get("/docs/theme", include_in_schema=False)
    def themed_docs_redirect():
        return RedirectResponse(url="/docs")

    return app


# Instantiate the app for ASGI servers like uvicorn
app = create_app()

# PUBLIC_INTERFACE
def get_app() -> FastAPI:
    """Expose the FastAPI app instance (for ASGI servers or tests)."""
    return app
