from fastapi import APIRouter

from src.domain_models.config import ProjectConfig

router = APIRouter()


@router.post("/config/submit", response_model=ProjectConfig)
def submit_config(config: ProjectConfig) -> ProjectConfig:
    """Accepts, validates, and translates a GUI configuration intent into a physical engine config."""
    # Since Pydantic validation runs automatically (including the @model_validator),
    # the returned config will have the properly translated hyperparameters.
    return config
