"""FastAPI entrypoint for the PlantGuard backend service."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import audio, fusion, image, text

app = FastAPI(title="PlantGuard Multimodal API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image.router, prefix="/vision", tags=["vision"])
app.include_router(audio.router, prefix="/speech", tags=["speech"])
app.include_router(text.router, prefix="/qa", tags=["qa"])
app.include_router(fusion.router, prefix="/multimodal", tags=["multimodal"])


@app.get("/health", tags=["system"])
async def healthcheck() -> dict[str, str]:
    """Simple readiness endpoint for orchestration and CI checks."""
    return {"status": "ok"}
