import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InterfaceTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element1: str = Field(..., description="First element or material (e.g., 'FePt')")
    element2: str = Field(..., description="Second element or material (e.g., 'MgO')")
    face1: str = Field(default="Fe", description="Terminating face of first material")
    face2: str = Field(default="Mg", description="Terminating face of second material")


def _secure_resolve_and_validate_dir(path_str: str, check_exists: bool = True) -> str | None:
    path = Path(path_str)

    if not path.is_absolute():
        msg = f"Directory path must be absolute: {path_str}"
        raise ValueError(msg)

    if check_exists:
        try:
            st_info = os.lstat(path)
            if stat.S_ISLNK(st_info.st_mode):
                msg = f"Directory {path_str} cannot be a symlink."
                raise ValueError(msg)
        except FileNotFoundError:
            logging.warning(f"Directory skipped as it does not exist: {path_str}")
            return None

    resolved = path.resolve(strict=check_exists)

    if check_exists and not resolved.is_dir():
        msg = f"Path must be a directory: {path_str}"
        raise ValueError(msg)

    restricted_prefixes = ["/etc", "/bin", "/usr", "/sbin", "/var", "/lib", "/boot", "/root"]
    for restricted in restricted_prefixes:
        if str(resolved).startswith(restricted):
            msg = f"Directory cannot be a system directory: {restricted}"
            raise ValueError(msg)

    if check_exists:
        st = resolved.stat()
        if bool(st.st_mode & stat.S_IWOTH):
            msg = f"Directory {path_str} is world-writable, which is insecure."
            raise ValueError(msg)

        if st.st_uid != os.getuid():
            msg = f"Directory {path_str} is not owned by the current user. This is insecure."
            raise ValueError(msg)

    return str(resolved)


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str] = Field(..., min_length=1, description="List of elements in the system")
    baseline_potential: Literal["lj", "zbl"] = Field(
        default="zbl", description="Baseline potential for core repulsion"
    )
    interface_target: InterfaceTarget | None = Field(
        default=None, description="Target interface structure configuration"
    )
    interface_generation_iteration: int = Field(
        default=0, description="The AL iteration number to inject the generated interface target."
    )
    restricted_directories: list[str] = Field(
        default_factory=lambda: ["/etc", "/bin", "/usr", "/sbin", "/var", "/lib", "/boot", "/root"],
        description="System directories forbidden from sandbox execution.",
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

    @field_validator("lmp_binary", "eon_binary")
    @classmethod
    def validate_binary_names(cls, v: str) -> str:
        if "/" in v or "\\" in v or ".." in v:
            msg = "Binary name cannot contain path separators or traversal characters."
            raise ValueError(msg)
        return v

    trusted_directories: list[str] = Field(
        ...,
        description="List of trusted directories for executables. Required.",
    )
    binary_hashes: dict[str, str] = Field(
        default={},
        description="Optional dict mapping binary names to SHA256 hashes for strict validation",
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
    box_size: list[int] = Field(default=[2, 2, 2], description="Supercell dimensions [x, y, z]")
    lattice_size: float = Field(default=3.5, gt=0.0, description="Lattice parameter constant")
    lattice_type: str = Field(default="fcc", description="Lattice system type string")
    eon_job: str = Field(
        default="process_search", description="EON Main Job configuration parameter"
    )
    eon_min_mode_method: str = Field(
        default="dimer", description="EON Process Search configuration parameter"
    )
    project_root: str = Field(
        ...,
        description="Project root directory for resolving binary paths. Required.",
    )

    @field_validator("project_root")
    @classmethod
    def validate_project_root_str(cls, v: str) -> str:
        res = _secure_resolve_and_validate_dir(v, check_exists=True)
        if res is None:
            msg = f"project_root skipped or invalid: {v}"
            raise ValueError(msg)
        return res

    @classmethod
    def validate_trusted_directories(cls, v: list[str]) -> list[str]:
        validated = []
        for path in v:
            res = _secure_resolve_and_validate_dir(path, check_exists=True)
            if res:
                validated.append(res)
        return validated

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
    max_potential_size: int = Field(
        default=104857600,
        ge=1024,
        description="Maximum allowed YACE file size in bytes (default 100MB)",
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

    @field_validator("pace_train_binary", "pace_activeset_binary")
    @classmethod
    def validate_binary_names(cls, v: str) -> str:
        if "/" in v or "\\" in v or ".." in v:
            msg = "Binary name cannot contain path separators or traversal characters."
            raise ValueError(msg)
        return v

    trusted_directories: list[str] = Field(
        ...,
        description="List of trusted directories for executables. Required.",
    )
    binary_hashes: dict[str, str] = Field(
        default={},
        description="Optional dict mapping binary names to SHA256 hashes for strict validation",
    )

    @field_validator("trusted_directories")
    @classmethod
    def validate_trusted_directories(cls, v: list[str]) -> list[str]:
        validated = []
        for path in v:
            res = _secure_resolve_and_validate_dir(path, check_exists=True)
            if res:
                validated.append(res)
        return validated


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
    valid_interface_targets: list[str] = Field(
        default_factory=lambda: ["FePt", "MgO"],
        description="Whitelist of explicitly supported synthetic interface components",
    )
    fept_lattice_constant: float = Field(default=3.8, description="Lattice constant for FePt")
    mgo_lattice_constant: float = Field(default=4.21, description="Lattice constant for MgO")
    interface_max_strain: float = Field(
        default=10.0, description="Max strain for stacking interfaces"
    )


class PolicyConfig(BaseModel):


    @field_validator("fallback_metal_melting_point", "fallback_insulator_melting_point")
    @classmethod
    def validate_melting_point(cls, v: float) -> float:
        if v <= 0.0:
            msg = "Melting point must be positive"
            raise ValueError(msg)
        return v

    @field_validator("fallback_metal_bulk_modulus", "fallback_insulator_bulk_modulus")
    @classmethod
    def validate_bulk_modulus(cls, v: float) -> float:
        if v <= 0.0:
            msg = "Bulk modulus must be positive"
            raise ValueError(msg)
        return v

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
    uncertainty_variance_threshold: float = Field(
        default=1.0, description="Threshold for initial gamma variance to trigger cautious policy"
    )
    metal_band_gap_threshold: float = Field(
        default=0.1, description="Band gap threshold below which a material is considered a metal"
    )
    hard_material_bulk_modulus_threshold: float = Field(
        default=200.0,
        description="Bulk modulus threshold above which a material is considered hard",
    )
    fallback_metal_band_gap: float = Field(default=0.0, description="Fallback band gap for metals")
    fallback_metal_melting_point: float = Field(
        default=1500.0, description="Fallback melting point for metals"
    )
    fallback_metal_bulk_modulus: float = Field(
        default=150.0, description="Fallback bulk modulus for metals"
    )
    fallback_insulator_band_gap: float = Field(
        default=2.0, description="Fallback band gap for insulators"
    )
    fallback_insulator_melting_point: float = Field(
        default=800.0, description="Fallback melting point for insulators"
    )
    fallback_insulator_bulk_modulus: float = Field(
        default=50.0, description="Fallback bulk modulus for insulators"
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
    use_mock: bool = Field(default=False, description="Enable mock mode for CI testing")

    @classmethod
    def _validate_env_key(cls, key: str) -> None:
        if not key.startswith("MLIP_"):
            msg = f"Unauthorized environment variable injected via .env: {key}. Only MLIP_ prefixes are allowed."
            raise ValueError(msg)
        if len(key) > 64:
            msg = "Environment variable key exceeds maximum length"
            raise ValueError(msg)
        if not re.match(r"^[A-Z0-9_]+$", key):
            msg = f"Invalid characters in .env variable key: {key}"
            raise ValueError(msg)

    @classmethod
    def _validate_env_value(cls, val: str) -> None:
        if len(val) > 1024:
            msg = "Environment variable value exceeds maximum length"
            raise ValueError(msg)
        if ".." in val or ";" in val or "&" in val or "|" in val:
            msg = f"Invalid characters or traversal sequences in .env variable value: {val}."
            raise ValueError(msg)

    @classmethod
    def _validate_env_file_security(cls, env_file: Path, expected_base: Path) -> Path:
        if env_file.is_symlink():
            msg = ".env file must not be a symlink."
            raise ValueError(msg)

        resolved_env = env_file.resolve(strict=True)

        if resolved_env.parent != expected_base.resolve(strict=True):
            msg = f".env file must reside directly in the allowed base directory: {expected_base}"
            raise ValueError(msg)

        st = os.lstat(resolved_env)

        if st.st_size > 10 * 1024:
            msg = ".env file exceeds maximum allowed size (10KB)."
            raise ValueError(msg)

        if st.st_uid != os.getuid():
            msg = ".env file is not owned by the current user."
            raise ValueError(msg)

        if bool(st.st_mode & stat.S_IRWXO) or bool(st.st_mode & stat.S_IRWXG):
            msg = ".env file has insecure permissions. It must not be group or world readable/writable."
            raise ValueError(msg)

        return resolved_env

    @model_validator(mode="before")
    @classmethod
    def validate_env_content(cls, values: dict[str, Any]) -> dict[str, Any]:
        expected_base = Path.cwd().resolve(strict=True)
        env_file = expected_base / ".env"

        if env_file.exists():
            if not env_file.is_file():
                msg = ".env must be a regular file."
                raise ValueError(msg)

            if env_file.is_symlink():
                msg = ".env file must not be a symlink."
                raise ValueError(msg)

            resolved_env = env_file.resolve(strict=True)

            # Additional content safety checks
            with Path.open(resolved_env, encoding="utf-8") as f:
                content = f.read()
                if "import " in content or "eval(" in content or "exec(" in content:
                    msg = "Suspicious content detected in .env file."
                    raise ValueError(msg)

            with Path.open(resolved_env, encoding="utf-8") as f:
                for raw_line in f:
                    clean_line = raw_line.strip()
                    if not clean_line or clean_line.startswith("#") or "=" not in clean_line:
                        continue

                    key, val = clean_line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')

                    cls._validate_env_key(key)
                    cls._validate_env_value(val)

        return values

    @field_validator("project_root", mode="before")
    @classmethod
    def convert_str_to_path(cls, v: str | Path) -> Path:
        p = Path(v)
        if ".." in str(p):
            msg = "Path traversal characters (..) are not allowed in project_root"
            raise ValueError(msg)
        if not p.is_absolute():
            msg = "project_root must be an absolute path"
            raise ValueError(msg)
        return p.resolve(strict=True)

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
        if ".." in str(v):
            msg = "Path traversal sequences (..) are not allowed in project_root"
            raise ValueError(msg)

        # Canonicalize path and resolve symlinks securely without os.path.realpath
        resolved_path = Path(v).resolve(strict=True)

        if not resolved_path.is_absolute():
            msg = f"Project root directory '{v}' must be an absolute path."
            raise ValueError(msg)

        if not resolved_path.exists() or not resolved_path.is_dir():
            msg = f"Project root directory '{v}' does not exist or is not a directory."
            raise ValueError(msg)

        # Security: forbid system directories
        from src.domain_models.config import SystemConfig

        restricted_prefixes = SystemConfig.model_fields["restricted_directories"].default_factory()  # type: ignore
        for restricted in restricted_prefixes:
            if str(resolved_path).startswith(restricted):
                msg = f"project_root cannot be a system directory: {restricted}"
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
