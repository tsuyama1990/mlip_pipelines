from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(title="Adaptive-MLIP API", description="API Gateway for Adaptive MLIP builder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
