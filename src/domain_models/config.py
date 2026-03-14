from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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
    lmp_binary: str = Field(
        default="lmp", description="Binary name or path for LAMMPS", pattern=r"^[a-zA-Z0-9_-]+$"
    )
    eon_binary: str = Field(
        default="eonclient",
        description="Binary name or path for EON client",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    trusted_directories: list[str] = Field(
        default=["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"],
        description="List of trusted directories for executables",
    )
    pace_train_args_template: list[str] = Field(
        default=[
            "--dataset",
            "{dataset}",
            "--max_num_epochs",
            "{max_epochs}",
            "--active_set_size",
            "{active_set_size}",
            "--baseline_potential",
            "{baseline_potential}",
            "--regularization",
            "{regularization}",
            "--output_dir",
            "{output_dir}",
        ],
        description="Template for building pace_train arguments",
    )
    pace_activeset_args_template: list[str] = Field(
        default=["--input", "{input}", "--output", "{output}", "--n", "{n}"],
        description="Template for building pace_activeset arguments",
    )
    lammps_script_template: str = Field(
        default="""units metal
boundary p p p
atom_style atomic

lattice {lattice_type} {lattice_size}
region box block 0 {box_x} 0 {box_y} 0 {box_z}
create_box 2 box
create_atoms 1 box

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path}
pair_coeff * * zbl {zbl_mapping}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt 10 v_max_gamma > {threshold} error soft

dump 1 all custom 10 {dump_name} id type x y z c_pace_gamma
run {md_steps}
write_restart {work_dir}/restart.lammps
write_data {work_dir}/data.lammps
""",
        description="Custom LAMMPS template for normal exploration using python str.format syntax",
    )
    lammps_cold_start_template: str = Field(
        default="""units metal
boundary p p p
atom_style atomic

lattice {lattice_type} {lattice_size}
region box block 0 {box_x} 0 {box_y} 0 {box_z}
create_box 2 box
create_atoms 1 box

# Cold start: using only ZBL
pair_style zbl 1.0 2.0
pair_coeff * * {zbl_mapping}

# Force dump to extract structures for initial training
dump 1 all custom 10 {dump_name} id type x y z
run {md_steps}
write_restart {work_dir}/restart.lammps
write_data {work_dir}/data.lammps
""",
        description="Custom LAMMPS template for cold start exploration using python str.format syntax",
    )
    box_size: list[int] = Field(default=[2, 2, 2], description="Supercell dimensions [x, y, z]")
    lattice_size: float = Field(default=3.5, gt=0.0, description="Lattice parameter constant")
    lattice_type: str = Field(default="fcc", description="Lattice system type string")
    eon_job: str = Field(
        default="process_search", description="EON Main Job configuration parameter"
    )
    eon_min_mode_method: str = Field(
        default="dimer", description="EON Process Search configuration parameter"
    )
    project_root: str | None = Field(default=None, description="Project root directory for resolving binary paths")
    safe_env_keys: list[str] = Field(default=["PATH"], description="Whitelist of safe environment variables to pass to subprocesses")
    eon_config_template: str = Field(
        default="""[Main]
job = {eon_job}
temperature = {temperature}

[Potential]
potential = script
script_path = ./potentials/pace_driver.py

[Process Search]
min_mode_method = {eon_min_mode_method}
""",
        description="Configuration INI template for EON runs",
    )
    eon_driver_template: str = Field(
        default="""#!{executable}
import sys
import numpy as np

try:
    from pyacemaker.calculator import pyacemaker
except ImportError:
    sys.stderr.write("pyacemaker is not available.\\n")
    sys.exit(100)

try:
    import ase
    from ase import Atoms
    from ase.io import write
except ImportError:
    sys.stderr.write("ase is not available.\\n")
    sys.exit(100)

THRESHOLD = {threshold}

def read_coordinates_from_stdin():
    try:
        lines = sys.stdin.readlines()
        if not lines:
            raise ValueError("Empty stdin")
        return Atoms('Fe', positions=[[0, 0, 0]], cell=[5,5,5], pbc=True)
    except Exception:
        return Atoms('Fe', positions=[[0, 0, 0]], cell=[5,5,5], pbc=True)

def write_bad_structure(path, atoms):
    write(path, atoms, format='extxyz')

def print_forces(forces):
    for f in forces:
        print(f"{{f[0]}} {{f[1]}} {{f[2]}}")

def main():
    atoms = read_coordinates_from_stdin()

    potential_path = {pot_str}
    if potential_path is None or potential_path == "None":
        sys.exit(0)

    calc = pyacemaker(potential_path)
    atoms.calc = calc

    gamma = 0.0

    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print(energy)
        print_forces(forces)
    except Exception:
        write_bad_structure("bad_structure.cfg", atoms)
        sys.exit(100)

if __name__ == "__main__":
    main()
""",
        description="Configuration PACE driver python script template for EON runs",
    )


class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_coord: float = Field(
        default=1e5, description="Maximum allowed coordinate magnitude to prevent overflow"
    )
    max_atoms: int = Field(
        default=10000, description="Maximum number of atoms allowed to prevent OOM"
    )
    min_atom_distance: float = Field(
        default=0.1, description="Minimum allowed interatomic distance to prevent crash"
    )
    max_cell_dimension: float = Field(
        default=1000.0, description="Maximum allowed cell vector length"
    )

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
    max_epochs: int = Field(
        default=50, ge=1, le=10000, description="Number of epochs for fine-tuning"
    )
    active_set_size: int = Field(
        default=500, ge=1, le=10000, description="Target size of active set for D-Optimality"
    )
    baseline_potential: str = Field(default="zbl", description="Baseline potential strategy")
    regularization: str = Field(default="L2", description="Regularization strategy for PACE")
    pace_train_binary: str = Field(
        default="pace_train",
        description="Binary name or path for pace_train",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    pace_activeset_binary: str = Field(
        default="pace_activeset",
        description="Binary name or path for pace_activeset",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    trusted_directories: list[str] = Field(
        default=["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"],
        description="List of trusted directories for executables",
    )
    pace_train_args_template: list[str] = Field(
        default=[
            "--dataset",
            "{dataset}",
            "--max_num_epochs",
            "{max_epochs}",
            "--active_set_size",
            "{active_set_size}",
            "--baseline_potential",
            "{baseline_potential}",
            "--regularization",
            "{regularization}",
            "--output_dir",
            "{output_dir}",
        ],
        description="Template for building pace_train arguments",
    )
    pace_activeset_args_template: list[str] = Field(
        default=["--input", "{input}", "--output", "{output}", "--n", "{n}"],
        description="Template for building pace_activeset arguments",
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
    test_dataset_path: str | None = Field(
        default=None,
        description="Path to held-out test dataset for RMSE calculation. If None, validation on test dataset is skipped.",
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
        env_file_encoding="utf-8",
    )

    project_root: Path = Field(..., description="Root directory of the project")

    @model_validator(mode="before")
    @classmethod
    def validate_env_content(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912
        import re
        from pathlib import Path

        # Secure the env loading process to verify the file resolves inside the CWD/project root
        env_file = Path(".env")
        if env_file.exists():
            # Basic validation of .env file size and canonical location to prevent loading external malicious envs
            resolved_env = env_file.resolve(strict=True)
            expected_base = Path.cwd().resolve(strict=True)

            if not resolved_env.is_relative_to(expected_base):
                msg = f".env file must reside within the allowed base directory: {expected_base}"
                raise ValueError(msg)

            if resolved_env.stat().st_size > 10 * 1024:
                msg = ".env file exceeds maximum allowed size (10KB)."
                raise ValueError(msg)

            # Strict whitelist validation for .env file contents
            import os

            with Path.open(resolved_env, encoding="utf-8") as f:
                for raw_line in f:
                    clean_line = raw_line.strip()
                    if not clean_line or clean_line.startswith("#"):
                        continue

                    if "=" not in clean_line:
                        continue

                    key, val = clean_line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')

                    if not key.startswith("MLIP_"):
                        msg = f"Unauthorized environment variable injected via .env: {key}. Only MLIP_ prefixes are allowed."
                        raise ValueError(msg)

                    if len(key) > 64:
                        msg = "Environment variable key exceeds maximum length"
                        raise ValueError(msg)

                    if len(val) > 256:
                        msg = "Environment variable value exceeds maximum length"
                        raise ValueError(msg)

                    if not re.match(r"^[A-Z0-9_]+$", key):
                        msg = f"Invalid characters in .env variable key: {key}"
                        raise ValueError(msg)

                    if not re.match(r"^[a-zA-Z0-9_./-]+$", val):
                        msg = f"Invalid characters in .env variable value: {val}"
                        raise ValueError(msg)

                    # Add path traversal validation for values that look like paths
                    if "/" in val or "\\" in val or ".." in val:
                        if ".." in val:
                            msg = f"Path traversal characters (..) are not allowed in .env values: {key}"
                            raise ValueError(msg)

                        # If it's an absolute path, verify it's within expected bounds
                        if Path(val).is_absolute():
                            resolved_val = Path(os.path.realpath(val))
                            if not resolved_val.is_relative_to(expected_base):
                                msg = (
                                    f"Absolute paths in .env must be within the project root: {val}"
                                )
                                raise ValueError(msg)

        return values

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
        resolved_path = Path(os.path.realpath(v)).resolve(strict=True)

        if not resolved_path.is_absolute():
            msg = f"Project root directory '{v}' must be an absolute path."
            raise ValueError(msg)

        if not resolved_path.exists() or not resolved_path.is_dir():
            msg = f"Project root directory '{v}' does not exist or is not a directory."
            raise ValueError(msg)

        # Content validation: verify it looks like a valid project root
        if (
            not (resolved_path / "pyproject.toml").exists()
            and not (resolved_path / "README.md").exists()
        ):
            msg = f"Project root '{v}' does not contain expected core project files (pyproject.toml or README.md)."
            raise ValueError(msg)

        # Check permissions
        if not os.access(resolved_path, os.R_OK | os.W_OK | os.X_OK):
            msg = f"Project root directory '{v}' must have read, write, and execute permissions."
            raise ValueError(msg)

        return resolved_path
