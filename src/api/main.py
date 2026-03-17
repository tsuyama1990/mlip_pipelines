from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="Adaptive-MLIP GUI Platform",
    description="Next-Generation GUI Platform API Gateway for Adaptive-MLIP",
    version="0.1.0",
)

# Standard CORS middleware for local React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
