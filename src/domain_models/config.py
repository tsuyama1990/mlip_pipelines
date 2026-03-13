from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str] = Field(..., min_length=1, description="List of elements in the system")
    baseline_potential: Literal["lj", "zbl"] = Field(
        default="zbl", description="Baseline potential for core repulsion"
    )


class DynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uncertainty_threshold: float = Field(
        default=5.0, ge=0.0, description="Gamma threshold to trigger halt"
    )
    md_steps: int = Field(default=100000, ge=1, description="Number of MD steps per exploration run")
    temperature: float = Field(default=300.0, ge=0.0, description="Temperature for MD exploration")
    pressure: float = Field(default=0.0, description="Pressure for NPT MD exploration")


class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kspacing: float = Field(default=0.05, gt=0.0, description="K-point spacing in inverse Angstroms")
    smearing_width: float = Field(default=0.02, ge=0.0, description="Smearing width (Ry)")
    pseudo_dir: str = Field(
        default=str(Path.home() / "pseudos"), description="Path to pseudopotentials directory (can be overridden by MLIP_PSEUDO_DIR env var)"
    )
    max_retries: int = Field(default=3, ge=0, description="Max retries for SCF convergence failure")
    buffer_size: float = Field(default=4.0, ge=0.0, description="Buffer size in Angstroms for periodic embedding")
    transition_metals: list[str] = Field(
        default=["Fe", "Co", "Ni", "Mn", "Cr", "V"], description="List of transition metals for spin-polarization heuristics"
    )


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_epochs: int = Field(default=50, ge=1, description="Number of epochs for fine-tuning")
    active_set_size: int = Field(
        default=500, ge=1, description="Target size of active set for D-Optimality"
    )
    baseline_potential: str = Field(default="zbl", description="Baseline potential strategy")
    regularization: str = Field(default="L2", description="Regularization strategy for PACE")


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_rmse_threshold: float = Field(default=0.002, ge=0.0, description="Max allowed Energy RMSE (eV/atom)")
    force_rmse_threshold: float = Field(default=0.05, ge=0.0, description="Max allowed Force RMSE (eV/A)")
    stress_rmse_threshold: float = Field(default=0.1, ge=0.0, description="Max allowed Stress RMSE (GPa)")


class StructureGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stdev: float = Field(default=0.05, ge=0.0, description="Standard deviation for rattling")
    seed_base: int = Field(default=42, description="Base seed for reproducibility")


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_t_max_scale: float = Field(default=0.5, ge=0.0, description="Default max temp scale vs melting point")
    cautious_t_max_scale: float = Field(default=0.3, ge=0.0, description="Cautious max temp scale vs melting point")
    high_mc_t_max_scale: float = Field(default=0.8, ge=0.0, description="High-MC max temp scale vs melting point")


class ProjectConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MLIP_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    project_root: Path = Field(..., description="Root directory of the project")
    system: SystemConfig
    dynamics: DynamicsConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    validator: ValidatorConfig
    structure_generator: StructureGeneratorConfig = Field(default_factory=StructureGeneratorConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
