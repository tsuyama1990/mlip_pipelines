from pydantic import BaseModel, ConfigDict, Field


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str]
    mass: dict[str, float] | None = None
    lattice_parameters: dict[str, float] | None = None
    structure_type: dict[str, str] | None = None


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
    energy_weight: float = Field(default=1.0, ge=0.0)
    forces_weight: float = Field(default=10.0, ge=0.0)


class DynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gamma_threshold: float = Field(default=5.0, gt=0.0)
    max_temperature_threshold: float = Field(default=500.0, gt=0.0)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: SystemConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    potential_path_template: str = Field(default="potentials/generation_{iteration:03d}.yace")
    data_directory: str = Field(default="potentials")
