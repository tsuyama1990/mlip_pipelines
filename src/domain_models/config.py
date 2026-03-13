from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(default=300.0, ge=0.0)
    steps: int = Field(default=10000, gt=0)


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kspacing: float = Field(default=0.05, gt=0.0)
    smearing: str = Field(default="mv")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_epochs: int = Field(default=50, gt=0)


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rmse_energy_threshold: float = Field(default=2.0, gt=0.0)
    rmse_force_threshold: float = Field(default=0.05, gt=0.0)


class OTFLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uncertainty_threshold: float = Field(default=5.0, gt=0.0)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str = Field(default="mlip_project")
    data_directory: Path = Field(default_factory=lambda: Path("data"))
    potential_path_template: str = Field(default="potentials/generation_{iteration:03d}.yace")
    lammps: MDConfig = Field(default_factory=MDConfig)
    dft: DFTConfig = Field(default_factory=DFTConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    otf_loop: OTFLoopConfig = Field(default_factory=OTFLoopConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**data)
