from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.dynamics.security_utils import _validate_string_security


class GUIStateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON-like dictionary for GUI state"
    )

class WorkflowIntentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_material: str = Field(..., description="Target material string")
    accuracy_speed_tradeoff: int = Field(..., ge=1, le=10, description="Trade-off slider from 1 (speed) to 10 (accuracy)")
    enable_auto_hpo: bool = Field(default=False, description="Enable Auto HPO")

    @field_validator("target_material")
    @classmethod
    def validate_target_material(cls, v: str) -> str:
        _validate_string_security(v)
        return v
