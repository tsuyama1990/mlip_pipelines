from pathlib import Path
from typing import Self

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MaterialFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str] = Field(..., description="Elements present in the material")
    band_gap: float = Field(default=0.0, ge=0.0, description="Estimated band gap in eV")
    bulk_modulus: float = Field(default=100.0, ge=0.0, description="Estimated bulk modulus in GPa")
    melting_point: float = Field(default=1000.0, gt=0.0, description="Estimated melting point in K")
    initial_gamma_variance: float = Field(
        default=0.1, ge=0.0, description="Variance of gamma in initial structure"
    )


class ExplorationStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    md_mc_ratio: float = Field(default=0.0, ge=0.0, description="Ratio of MD to MC steps")
    t_max: float = Field(default=300.0, ge=0.0, description="Maximum temperature for schedule")
    n_defects: float = Field(default=0.0, ge=0.0, description="Density of defects to introduce")
    strain_range: float = Field(
        default=0.0, ge=0.0, description="Range of strain to apply (e.g. 0.15)"
    )
    policy_name: str = Field(..., description="Name of the decided policy")


class HaltInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    halted: bool = Field(..., description="Whether the simulation halted due to uncertainty")
    dump_file: Path | None = Field(default=None, description="Path to the LAMMPS dump file at halt")
    high_gamma_atoms: list[Atoms] | None = Field(
        default=None, description="Extracted high gamma environments"
    )
    max_gamma: float | None = Field(default=None, description="Max gamma recorded at halt")

    @model_validator(mode="after")
    def validate_halted_state(self) -> Self:
        if self.halted and self.high_gamma_atoms is None:
            msg = "high_gamma_atoms must not be None when halted is True"
            raise ValueError(msg)
        return self


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool = Field(..., description="Whether validation overall passed")
    reason: str | None = Field(default=None, description="Reason if validation failed")
    energy_rmse: float = Field(..., description="Energy RMSE in eV/atom")
    force_rmse: float = Field(..., description="Force RMSE in eV/A")
    stress_rmse: float = Field(..., description="Stress RMSE in GPa")
    phonon_stable: bool = Field(
        ..., description="Whether phonons are stable (no imaginary frequencies)"
    )
    mechanically_stable: bool = Field(..., description="Whether it meets Born criteria")


class ValidationScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rmse_energy: float = Field(..., description="Energy RMSE in eV/atom")
    rmse_forces: float = Field(..., description="Force RMSE in eV/A")
    rmse_stress: float = Field(..., description="Stress RMSE in GPa")
    born_stable: bool = Field(..., description="Whether it passes Born stability criteria")
    phonon_stable: bool = Field(..., description="Whether it has no imaginary phonon frequencies")


class CutoutResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    cluster: Atoms = Field(..., description="The extracted ASE Atoms cluster")
    passivation_atoms_added: int = Field(
        default=0, ge=0, description="Number of passivation atoms added"
    )
