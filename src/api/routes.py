from fastapi import APIRouter

from src.domain_models.config import ProjectConfig

router = APIRouter()


@router.post("/config/submit", response_model=ProjectConfig)
async def submit_config(config: ProjectConfig) -> ProjectConfig:
    """
    Receives a fully constructed or partially intent-driven ProjectConfig from the GUI.
    Pydantic automatically validates security constraints and executes the intent
    mathematical translations. Returns the fully resolved configuration.
    """
    return config
