import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.dynamics.security_utils import _validate_string_security


class GUIStateConfig(BaseModel):
    """Presentational state schema for the frontend."""
    model_config = ConfigDict(extra="forbid")

    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON-like dictionary for React Flow data"
    )

    @field_validator("state")
    @classmethod
    def validate_state_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(v)) > 1048576:
            raise ValueError("GUIStateConfig state exceeds maximum allowed size of 1MB")
        return v

class WorkflowIntentConfig(BaseModel):
    """High-level user intent configuration."""
    model_config = ConfigDict(extra="forbid")

    target_material: str = Field(..., description="Target material string")
    accuracy_speed_tradeoff: int = Field(
        ..., ge=1, le=10, description="Accuracy vs. Speed tradeoff slider (1-10)"
    )
    enable_auto_hpo: bool = Field(default=False, description="Enable Auto HPO")

    @field_validator("target_material")
    @classmethod
    def validate_target_material(cls, v: str) -> str:
        _validate_string_security(v)
        return v
