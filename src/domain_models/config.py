from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    md_steps: int = Field(
        default=100000, ge=1, description="Number of MD steps per exploration run"
    )
    temperature: float = Field(default=300.0, ge=0.0, description="Temperature for MD exploration")
    pressure: float = Field(default=0.0, description="Pressure for NPT MD exploration")
    lmp_binary: str = Field(default="lmp", description="Binary name or path for LAMMPS")
    lammps_script_template: str | None = Field(
        default=None, description="Optional custom LAMMPS template using python str.format syntax"
    )


class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kspacing: float = Field(
        default=0.05, gt=0.0, description="K-point spacing in inverse Angstroms"
    )
    smearing_width: float = Field(default=0.02, ge=0.0, description="Smearing width (Ry)")
    pseudo_dir: str = Field(
        default=str(Path.home() / "pseudos"),
        description="Path to pseudopotentials directory (can be overridden by MLIP_PSEUDO_DIR env var)",
    )

    @field_validator("kspacing")
    @classmethod
    def validate_kspacing(cls, v: float) -> float:
        if v < 0.01 or v > 0.15:
            msg = "kspacing must be within a reasonable range for typical DFT calculations (0.01 to 0.15)"
            raise ValueError(msg)
        return v

    max_retries: int = Field(default=3, ge=0, description="Max retries for SCF convergence failure")
    buffer_size: float = Field(
        default=4.0, ge=0.0, description="Buffer size in Angstroms for periodic embedding"
    )
    transition_metals: list[str] = Field(
        default=["Fe", "Co", "Ni", "Mn", "Cr", "V"],
        description="List of transition metals for spin-polarization heuristics",
    )
    calculation: str = Field(default="scf", description="QE calculation type")
    ecutwfc: float = Field(default=40.0, ge=0.0, description="Wavefunction cutoff")
    ecutrho: float = Field(default=320.0, ge=0.0, description="Charge density cutoff")
    occupations: str = Field(default="smearing", description="Occupations setting")
    smearing: str = Field(default="mv", description="Smearing type")
    mixing_beta: float = Field(default=0.7, ge=0.0, le=1.0, description="Mixing beta")
    diagonalization: str = Field(default="david", description="Diagonalization algorithm")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_epochs: int = Field(default=50, ge=1, description="Number of epochs for fine-tuning")
    active_set_size: int = Field(
        default=500, ge=1, description="Target size of active set for D-Optimality"
    )
    baseline_potential: str = Field(default="zbl", description="Baseline potential strategy")
    regularization: str = Field(default="L2", description="Regularization strategy for PACE")
    pace_train_binary: str = Field(
        default="pace_train", description="Binary name or path for pace_train"
    )
    pace_activeset_binary: str = Field(
        default="pace_activeset", description="Binary name or path for pace_activeset"
    )


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_rmse_threshold: float = Field(
        default=0.002, ge=0.0, description="Max allowed Energy RMSE (eV/atom)"
    )
    force_rmse_threshold: float = Field(
        default=0.05, ge=0.0, description="Max allowed Force RMSE (eV/A)"
    )
    stress_rmse_threshold: float = Field(
        default=0.1, ge=0.0, description="Max allowed Stress RMSE (GPa)"
    )
    validation_element: str = Field(
        default="Fe", description="Element to build bulk validation structure from"
    )
    validation_crystal: str = Field(
        default="bcc", description="Crystal type to build bulk validation structure from"
    )
    validation_a: float = Field(default=2.86, description="Lattice parameter a")
    fallback_energy_rmse: float = Field(
        default=0.001, description="Fallback energy RMSE for mock CI environments"
    )
    fallback_force_rmse: float = Field(
        default=0.01, description="Fallback force RMSE for mock CI environments"
    )
    fallback_stress_rmse: float = Field(
        default=0.05, description="Fallback stress RMSE for mock CI environments"
    )


class StructureGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stdev: float = Field(default=0.05, ge=0.0, description="Standard deviation for rattling")
    seed_base: int = Field(default=42, description="Base seed for reproducibility")


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_t_max_scale: float = Field(
        default=0.5, ge=0.0, description="Default max temp scale vs melting point"
    )
    cautious_t_max_scale: float = Field(
        default=0.3, ge=0.0, description="Cautious max temp scale vs melting point"
    )
    high_mc_t_max_scale: float = Field(
        default=0.8, ge=0.0, description="High-MC max temp scale vs melting point"
    )
    default_md_mc_ratio: float = Field(default=0.0, description="Default md_mc_ratio")
    default_n_defects: float = Field(default=0.0, description="Default n_defects")
    default_strain_range: float = Field(default=0.0, description="Default strain_range")
    high_mc_ratio: float = Field(default=100.0, description="MD/MC ratio for High-MC policy")
    defect_driven_n_defects: float = Field(
        default=0.05, description="n_defects for Defect-Driven policy"
    )
    strain_heavy_range: float = Field(
        default=0.15, description="strain_range for Strain-Heavy policy"
    )


class ProjectConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MLIP_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    project_root: Path = Field(..., description="Root directory of the project")

    @field_validator("project_root", mode="before")
    @classmethod
    def convert_str_to_path(cls, v: str | Path) -> Path:
        return Path(v)

    system: SystemConfig
    dynamics: DynamicsConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    validator: ValidatorConfig
    structure_generator: StructureGeneratorConfig = Field(default_factory=StructureGeneratorConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    @field_validator("project_root")
    @classmethod
    def validate_project_root(cls, v: Path) -> Path:
        import os

        # Canonicalize path and resolve symlinks
        resolved_path = v.resolve(strict=False)

        if not resolved_path.is_absolute():
            msg = f"Project root directory '{v}' must be an absolute path."
            raise ValueError(msg)

        if ".." in str(resolved_path):
            msg = "Project root directory must not contain directory traversal characters."
            raise ValueError(msg)

        if not resolved_path.exists() or not resolved_path.is_dir():
            msg = f"Project root directory '{v}' does not exist or is not a directory."
            raise ValueError(msg)

        # Check permissions
        if not os.access(resolved_path, os.R_OK | os.W_OK | os.X_OK):
            msg = f"Project root directory '{v}' must have read, write, and execute permissions."
            raise ValueError(msg)

        return resolved_path
