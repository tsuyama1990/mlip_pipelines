import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from src.domain_models.config import ProjectConfig

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Adaptive-MLIP Configuration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/config/submit", response_model=ProjectConfig)
def submit_config(config: ProjectConfig) -> ProjectConfig:
    """
    Receives a ProjectConfig payload, validates it, and performs intent translation.
    Returns the fully resolved ProjectConfig.
    """
    return config
