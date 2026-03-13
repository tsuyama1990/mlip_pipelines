
from pydantic import BaseModel, ConfigDict


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str]
    mass: dict[str, float] | None = None

class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k_spacing: float
    pseudo_paths: dict[str, str]
    mixing_beta: float = 0.7
    smearing: float = 0.01

class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ace_max_degree: int
    lj_baseline_params: dict[str, float]

class DynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gamma_threshold: float = 5.0

class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: SystemConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
