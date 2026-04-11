import logging
import warnings
from contextlib import asynccontextmanager

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api.websocket import handle_session
from app.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SynthioLabs Voice Slide Deck backend starting up")
    yield
    logger.info("Backend shutting down")


app = FastAPI(
    title="SynthioLabs Voice Slide Deck",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/slides")
async def get_slides():
    """Return all slide metadata (used by frontend on load)."""
    from app.slides.content import SLIDES
    return [
        {"index": s.index, "title": s.title, "bullets": s.bullets}
        for s in SLIDES
    ]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_session(websocket)
