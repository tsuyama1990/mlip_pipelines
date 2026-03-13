from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MaterialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str] = Field(default_factory=lambda: ["Fe", "Pt"])
    atomic_numbers: list[int] = Field(default_factory=lambda: [26, 78])
    band_gap: float = Field(default=0.0, ge=0.0)
    melting_point: float = Field(default=1500.0, gt=0.0)
    bulk_modulus: float = Field(default=180.0, gt=0.0)
    crystal: str = Field(default="bcc")
    a: float = Field(default=2.8665)


class MDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(default=300.0, ge=0.0)
    steps: int = Field(default=10000, gt=0)
    lammps_script: list[str] = Field(
        default_factory=lambda: [
            "units metal",
            "boundary p p p",
            "atom_style atomic",
            "lattice bcc 2.8665",
            "region box block 0 2 0 2 0 2",
            "create_box 2 box",
            "create_atoms 1 box",
            "mass 1 55.845",
            "mass 2 195.084",
        ]
    )


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kspacing: float = Field(default=0.05, gt=0.0)
    smearing: str = Field(default="mv")
    ecutwfc: float = Field(default=40.0, gt=0.0)
    ecutrho: float = Field(default=320.0, gt=0.0)
    degauss: float = Field(default=0.02, ge=0.0)
    mixing_beta: float = Field(default=0.7, gt=0.0)
    electron_maxstep: int = Field(default=100, gt=0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_epochs: int = Field(default=50, gt=0)


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rmse_energy_threshold: float = Field(default=2.0, gt=0.0)
    rmse_force_threshold: float = Field(default=0.05, gt=0.0)
    supercell_matrix: list[list[int]] = Field(
        default_factory=lambda: [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    displacement_distance: float = Field(default=0.01, gt=0.0)
    mesh: list[int] = Field(default_factory=lambda: [20, 20, 20])


class OTFLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uncertainty_threshold: float = Field(default=5.0, gt=0.0)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str = Field(default="mlip_project")
    data_directory: Path = Field(default_factory=lambda: Path("data"))
    potential_path_template: str = Field(default="potentials/generation_{iteration:03d}.yace")
    material: MaterialConfig = Field(default_factory=MaterialConfig)
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
