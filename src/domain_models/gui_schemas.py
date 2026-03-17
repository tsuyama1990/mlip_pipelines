from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.dynamics.security_utils import _validate_string_security


class GUIStateConfig(BaseModel):
    """Purely presentational, stateless schema designed to store the positions,
    zoom levels, and connections of the React Flow DAG nodes."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[dict[str, Any]] = Field(default_factory=list, description="React Flow nodes")
    edges: list[dict[str, Any]] = Field(default_factory=list, description="React Flow edges")
    viewport: dict[str, Any] = Field(default_factory=dict, description="React Flow viewport state")

    @model_validator(mode="after")
    def validate_size(self) -> "GUIStateConfig":
        if len(self.nodes) > 1000:
            msg = "Too many nodes in GUIStateConfig. Limit is 1000."
            raise ValueError(msg)
        if len(self.edges) > 10000:
            msg = "Too many edges in GUIStateConfig. Limit is 10000."
            raise ValueError(msg)
        return self


class WorkflowIntentConfig(BaseModel):
    """Captures the core scientific objective of the user."""

    model_config = ConfigDict(extra="forbid")

    target_material: str = Field(..., description="Target material (e.g., Pt-Ni)")
    accuracy_speed_tradeoff: int = Field(
        ..., ge=1, le=10, description="Tradeoff slider from 1 (Speed) to 10 (Accuracy)"
    )
    enable_auto_hpo: bool = Field(default=False, description="Enable Bayesian optimization paths")

    @model_validator(mode="after")
    def validate_target_material(self) -> "WorkflowIntentConfig":
        _validate_string_security(self.target_material)
        return self
