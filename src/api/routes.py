from fastapi import APIRouter

from src.domain_models.config import ProjectConfig

router = APIRouter()

@router.post("/config/submit")
async def submit_config(config: ProjectConfig) -> ProjectConfig:
    return config
