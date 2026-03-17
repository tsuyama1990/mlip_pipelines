import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.dynamics.security_utils import _validate_string_security


class GUIStateConfig(BaseModel):
    """Configuration for storing raw React Flow state data."""

    model_config = ConfigDict(extra="forbid")

    flow_data: dict[str, Any] = Field(
        default_factory=dict, description="Raw JSON data representing the DAG UI state."
    )

    @field_validator("flow_data")
    @classmethod
    def validate_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(v)) >= 1048576:
            msg = "GUIStateConfig payload is too large, exceeding 1MB limit."
            raise ValueError(msg)
        return v


class WorkflowIntentConfig(BaseModel):
    """High-level user intent to be translated into actual hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    target_material: str = Field(..., description="Target material system (e.g., 'Pt-Ni')")
    accuracy_speed_tradeoff: int = Field(
        ...,
        ge=1,
        le=10,
        description="Tradeoff slider: 1 (Max Speed) to 10 (Max Accuracy)",
    )
    enable_auto_hpo: bool = Field(
        default=False, description="Enable automatic hyperparameter optimization."
    )

    @field_validator("target_material")
    @classmethod
    def check_security(cls, v: str) -> str:
        _validate_string_security(v)
        return v
