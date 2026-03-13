from typing import Any

from pydantic import BaseModel, ConfigDict


class ExplorationStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r_md_mc: float
    t_schedule: list[float]
    n_defects: int
    strain_range: float


class HaltEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestep: int
    max_gamma: float
    triggering_atoms_indices: list[int]
    halt_structure: Any  # ase Atoms


class ValidationScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rmse_energy: float
    rmse_force: float
    phonon_stable: bool
    born_criteria_passed: bool
