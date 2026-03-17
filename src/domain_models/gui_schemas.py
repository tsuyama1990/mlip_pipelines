import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.dynamics.security_utils import validate_string_security


class GUIStateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary JSON-like GUI state for React Flow"
    )

    @field_validator("state")
    @classmethod
    def validate_state_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(v)) >= 1048576:
            msg = "GUI state payload exceeds maximum allowed size of 1MB."
            raise ValueError(msg)
        return v


class WorkflowIntentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_material: str = Field(..., description="Target material for the MLIP (e.g., 'Pt-Ni')")
    accuracy_speed_tradeoff: int = Field(
        ..., ge=1, le=10, description="Tradeoff between accuracy (10) and speed (1)"
    )
    enable_auto_hpo: bool = Field(
        default=False, description="Enable automated hyperparameter optimization"
    )

    @field_validator("target_material")
    @classmethod
    def validate_target_material_security(cls, v: str) -> str:
        validate_string_security(v)
        return v
