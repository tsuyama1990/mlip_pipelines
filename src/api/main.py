from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="Adaptive-MLIP Configuration API",
    description="API for parsing and translating intent-driven GUI configuration for the Adaptive-MLIP backend.",
    version="0.1.0",
)

# Standard CORS middleware for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
