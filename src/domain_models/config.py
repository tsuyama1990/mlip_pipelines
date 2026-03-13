from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MaterialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str]
    atomic_numbers: list[int]
    masses: list[float]
    band_gap: float
    melting_point: float
    bulk_modulus: float
    crystal: str
    a: float


class MDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(default=300.0, ge=0.0)
    steps: int = Field(default=10000, gt=0)
    lammps_commands: list[str] = Field(default_factory=list)


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kspacing: float = Field(default=0.05, gt=0.0)
    smearing_type: str = Field(default="mv")
    occupations: str = Field(default="smearing")
    calculation: str = Field(default="scf")
    ecutwfc: float = Field(default=40.0, gt=0.0)
    ecutrho: float = Field(default=320.0, gt=0.0)
    degauss: float = Field(default=0.02, ge=0.0)
    mixing_beta: float = Field(default=0.7, gt=0.0)
    electron_maxstep: int = Field(default=100, gt=0)
    pseudopotentials: dict[str, str] = Field(default_factory=dict)
    pw_executable: str = Field(default="pw.x")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_epochs: int = Field(default=50, gt=0)
    pace_train_args: list[str] = Field(default_factory=list)
    pace_activeset_executable: str = Field(default="pace_activeset")
    pace_train_executable: str = Field(default="pace_train")
    activeset_fallback_strategy: str = Field(default="random")


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rmse_energy_threshold: float = Field(default=2.0, gt=0.0)
    rmse_force_threshold: float = Field(default=0.05, gt=0.0)
    rmse_energy_offset: float = Field(default=0.0)
    rmse_force_offset: float = Field(default=0.0)
    supercell_matrix: list[list[int]] = Field(
        default_factory=lambda: [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    displacement_distance: float = Field(default=0.01, gt=0.0)
    mesh: list[int] = Field(default_factory=lambda: [20, 20, 20])
    imaginary_freq_threshold: float = Field(default=-0.1)


class OTFLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uncertainty_threshold: float = Field(default=5.0, gt=0.0)


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metal_eg_threshold: float = Field(default=0.1)
    hard_b0_threshold: float = Field(default=150.0)

    high_mc_r_md_mc: int = Field(default=100)
    high_mc_t_max_ratio: float = Field(default=0.8)
    high_mc_steps: int = Field(default=20000)
    high_mc_defects: int = Field(default=1)
    high_mc_strain: float = Field(default=0.05)

    defect_r_md_mc: int = Field(default=0)
    defect_t_max_ratio: float = Field(default=0.5)
    defect_steps: int = Field(default=10000)
    defect_defects: int = Field(default=3)
    defect_strain: float = Field(default=0.02)

    strain_r_md_mc: int = Field(default=0)
    strain_t_max: float = Field(default=500.0)
    strain_steps: int = Field(default=10000)
    strain_defects: int = Field(default=0)
    strain_strain: float = Field(default=0.15)

    std_r_md_mc: int = Field(default=0)
    std_t_max: float = Field(default=300.0)
    std_steps: int = Field(default=10000)
    std_defects: int = Field(default=0)
    std_strain: float = Field(default=0.0)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str = Field(default="mlip_project")
    data_directory: Path = Field(default_factory=lambda: Path("data"))
    active_learning_dir: Path = Field(default_factory=lambda: Path("active_learning"))
    potential_path_template: str = Field(default="potentials/generation_{iteration:03d}.yace")
    material: MaterialConfig
    lammps: MDConfig = Field(default_factory=MDConfig)
    dft: DFTConfig = Field(default_factory=DFTConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    otf_loop: OTFLoopConfig = Field(default_factory=OTFLoopConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        from pydantic import ValidationError

        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            msg = f"YAML file {path_obj} must contain a top-level dictionary."
            raise TypeError(msg)

        try:
            return cls.model_validate(data)
        except ValidationError as e:
            msg = f"Invalid configuration file {path_obj}:\n{e}"
            raise ValueError(msg) from e
