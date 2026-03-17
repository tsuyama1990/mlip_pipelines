from typing import Any

from fastapi import APIRouter

from src.domain_models.config import ProjectConfig

router = APIRouter()

@router.post("/config/submit", response_model=dict[str, Any])
async def submit_config(config: ProjectConfig) -> dict[str, Any]:
    """
    Accepts a ProjectConfig payload from the GUI.
    The payload is strictly validated via Pydantic.
    Any WorkflowIntentConfig included triggers mathematical translation into underlying hyperparams.
    """
    # Because FastAPI successfully parsed it into the `config` argument, Pydantic validation
    # (including our Intent translation) has completely succeeded. We just return it.

    return {
        "status": "success",
        "message": "Configuration successfully validated and translated.",
        "config": config.model_dump(mode="json")
    }
