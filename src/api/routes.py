from fastapi import APIRouter

from src.domain_models.config import ProjectConfig

router = APIRouter()

@router.post("/config/submit")
def submit_config(config: ProjectConfig) -> dict:
    """Validate and return the fully resolved configuration object."""
    return {"message": "Configuration successfully validated", "config": config.model_dump(mode="json")}
