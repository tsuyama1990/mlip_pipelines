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


def _check_allowed_base_dirs(resolved_str: str, path_str: str) -> None:
    import tempfile

    # Use resolve(strict=True) on allowed bases
    home_dir = str(Path.home().resolve(strict=True))
    tmp_dir = str(Path(tempfile.gettempdir()).resolve(strict=True))

    # Explicitly check for /app mapping because the tests run within an /app container block
    app_dir = "/app"

    is_safe = False
    for safe_base in [home_dir, tmp_dir, app_dir]:
        try:
            if os.path.commonpath([safe_base, resolved_str]) == safe_base:
                is_safe = True
                break
        except ValueError:
            pass

    if not is_safe:
        msg = f"Directory {path_str} must reside securely within an allowed base directory (home, tmp, or /app)."
        raise ValueError(msg)

    restricted_prefixes = ["/etc", "/bin", "/usr", "/sbin", "/var", "/lib", "/boot", "/root"]
    for restricted in restricted_prefixes:
        try:
            is_restricted = os.path.commonpath([restricted, resolved_str]) == restricted
        except ValueError:
            continue
        if is_restricted:
            msg = f"Directory cannot be a system directory: {restricted}"
            raise ValueError(msg)


def _check_ownership_and_perms(resolved: Path, path_str: str) -> None:
    st = resolved.stat()
    if st.st_uid != os.getuid():
        msg = f"Directory {path_str} is not owned by the current user (uid={os.getuid()}). This is insecure."
        raise ValueError(msg)

    if bool(st.st_mode & stat.S_IWOTH):
        msg = f"Directory {path_str} is world-writable, which is insecure."
        raise ValueError(msg)


def _secure_resolve_and_validate_dir(path_str: str, check_exists: bool = True) -> str:
    # Use realpath for canonicalization
    import os

    canonical_path = Path(os.path.realpath(path_str))

    if not canonical_path.is_absolute():
        msg = f"Directory path must be absolute: {path_str}"
        raise ValueError(msg)

    if check_exists:
        if not canonical_path.exists():
            msg = f"Directory does not exist: {path_str}"
            raise ValueError(msg)

        if not canonical_path.is_dir():
            msg = f"Path must be a directory: {path_str}"
            raise ValueError(msg)

    _check_allowed_base_dirs(str(canonical_path), path_str)

    if check_exists:
        _check_ownership_and_perms(canonical_path, path_str)

    return str(canonical_path)


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


class ActiveLearningThresholds(BaseModel):
    """Two-tier thresholds inspired by FLARE."""

    model_config = ConfigDict(extra="forbid")
    threshold_call_dft: float = Field(
        default=0.05, description="Global threshold to halt MD and call DFT"
    )
    threshold_add_train: float = Field(
        default=0.02, description="Local threshold to select atoms for training"
    )
    smooth_steps: int = Field(
        default=3, description="Number of consecutive steps exceeding threshold required to halt"
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ActiveLearningThresholds":
        if self.threshold_call_dft < self.threshold_add_train:
            msg = (
                f"Global halt threshold ({self.threshold_call_dft}) must be strictly greater than "
                f"or equal to local training addition threshold ({self.threshold_add_train}) "
                "to prevent infinite loops."
            )
            raise ValueError(msg)
        if self.smooth_steps <= 0:
            msg = "smooth_steps must be strictly greater than zero"
            raise ValueError(msg)
        return self


class DynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uncertainty_threshold: float = Field(
        default=5.0,
        ge=0.0,
        description="Deprecated: use thresholds.threshold_call_dft instead. Retained for backward compatibility.",
    )
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)

    @model_validator(mode="after")
    def sync_thresholds(self) -> "DynamicsConfig":
        if self.uncertainty_threshold != 5.0 and self.thresholds.threshold_call_dft == 0.05:
            # If the user explicitly set the deprecated uncertainty_threshold but not the new thresholds
            self.thresholds.threshold_call_dft = self.uncertainty_threshold
        elif self.thresholds.threshold_call_dft != 0.05:
            # Sync back just in case old code reads uncertainty_threshold directly
            self.uncertainty_threshold = self.thresholds.threshold_call_dft
        return self

    md_steps: int = Field(
        default=100000, ge=1, description="Number of MD steps per exploration run"
    )
    temperature: float = Field(default=300.0, ge=0.0, description="Temperature for MD exploration")
    pressure: float = Field(default=0.0, description="Pressure for NPT MD exploration")
    lmp_binary: str = Field(
        default="lmp", description="Binary name or path for LAMMPS", pattern=r"^[/a-zA-Z0-9_.-]+$"
    )
    eon_binary: str = Field(
        default="eonclient",
        description="Binary name or path for EON client",
        pattern=r"^[/a-zA-Z0-9_.-]+$",
    )

    @field_validator("lmp_binary", "eon_binary")
    @classmethod
    def validate_binary_names(cls, v: str) -> str:
        if "/" in v or "\\" in v or ".." in v:
            msg = "Binary name cannot contain path separators or traversal characters."
            raise ValueError(msg)

        # Verify the binary is natively reachable in the system PATH
        import shutil

        if not shutil.which(v):
            msg = f"Binary {v} not found in PATH or not executable."
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

    @model_validator(mode="after")
    def validate_binary_hashes(self) -> "DynamicsConfig":
        import hashlib
        import shutil

        # Helper for validating a specific binary mapping
        def verify_hash(binary_name: str, expected_hash: str) -> None:
            bin_path_str = shutil.which(binary_name)
            if not bin_path_str:
                msg = f"Binary {binary_name} could not be resolved."
                raise ValueError(msg)

            bin_path = Path(bin_path_str)
            # Read in chunks to prevent OOM
            sha256 = hashlib.sha256()
            with Path.open(bin_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)

            if sha256.hexdigest() != expected_hash:
                msg = f"Binary {binary_name} hash mismatch. Expected {expected_hash}, got {sha256.hexdigest()}"
                raise ValueError(msg)

        # Check the explicitly required ones if hashes are provided
        if self.lmp_binary in self.binary_hashes:
            verify_hash(self.lmp_binary, self.binary_hashes[self.lmp_binary])

        if self.eon_binary in self.binary_hashes:
            verify_hash(self.eon_binary, self.binary_hashes[self.eon_binary])

        return self

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
    max_kpoints: int = Field(
        default=1000, gt=0, description="Maximum allowed k-points grid size to prevent OOM"
    )
    min_cell_dimension: float = Field(
        default=1e-5,
        gt=0.0,
        description="Minimum allowed cell dimension to prevent division by zero",
    )
    smearing_width: float = Field(default=0.02, ge=0.0, description="Smearing width (Ry)")
    pseudo_dir: str = Field(
        default=str(Path.home() / "pseudos"),
        description="Path to pseudopotentials directory (can be overridden by MLIP_PSEUDO_DIR env var)",
    )

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
    timeout: int = Field(
        default=3600, ge=1, description="Timeout in seconds for pace_train execution"
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
    mace_train_binary: str = Field(
        default="mace_run_train",
        description="Binary name or path for mace_run_train",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    mace_finetuning_epochs: int = Field(
        default=5, ge=1, le=1000, description="Number of epochs for MACE finetuning"
    )
    mace_learning_rate: float = Field(
        default=0.001, gt=0.0, description="Learning rate for MACE finetuning"
    )

    @field_validator("pace_train_binary", "pace_activeset_binary", "mace_train_binary")
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


def _validate_env_key(key: str) -> None:
    if not key.startswith("MLIP_"):
        msg = f"Unauthorized environment variable injected via .env: {key}. Only MLIP_ prefixes are allowed."
        raise ValueError(msg)
    if len(key) > 64:
        msg = "Environment variable key exceeds maximum length"
        raise ValueError(msg)
    if not re.match(r"^[A-Z0-9_]+$", key):
        msg = f"Invalid characters in .env variable key: {key}"
        raise ValueError(msg)


def _validate_env_value(val: str) -> None:
    if len(val) > 1024:
        msg = "Environment variable value exceeds maximum length"
        raise ValueError(msg)

    # Strictly whitelist allowed characters to prevent shell/JSON injection
    # Allow alphanumeric, basic punctuation, paths, and decimals
    if not re.match(r"^[-a-zA-Z0-9_.:/=,+]*$", val):
        msg = "Invalid characters detected in .env variable value."
        raise ValueError(msg)


def _validate_env_file_security(env_file: Path, expected_base: Path) -> Path:
    st = os.lstat(env_file)
    expected_base_resolved = expected_base.resolve(strict=True)

    if stat.S_ISLNK(st.st_mode):
        msg = ".env file must not be a symlink."
        raise ValueError(msg)

    resolved_env = env_file.resolve(strict=True)

    # Verify the canonicalized path sits within the trusted base directory
    if not str(resolved_env).startswith(str(expected_base_resolved)):
        msg = f".env file must reside within the allowed base directory: {expected_base}"
        raise ValueError(msg)

    st = os.lstat(resolved_env)

    if st.st_size > 1024:
        msg = ".env file exceeds maximum allowed size (1KB)."
        raise ValueError(msg)

    if st.st_uid != os.getuid():
        msg = ".env file is not owned by the current user."
        raise ValueError(msg)

    if bool(st.st_mode & stat.S_IRWXO) or bool(st.st_mode & stat.S_IRWXG):
        msg = ".env file has insecure permissions. It must not be group or world readable/writable."
        raise ValueError(msg)

    # Content validation for malicious patterns before parsing
    with Path.open(resolved_env, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Simple check to see if the key starts with MLIP_ and lacks injection symbols
            if "=" in stripped:
                key, val = stripped.split("=", 1)
                key = key.strip()
                if not key.startswith("MLIP_"):
                    msg = f"All .env file keys must start with 'MLIP_'. Found invalid key: {key}"
                    raise ValueError(msg)

                val = val.strip()
                if (
                    ".." in val
                    or ";" in val
                    or "&" in val
                    or "|" in val
                    or "$" in val
                    or "`" in val
                    or "{" in val
                    or "}" in val
                ):
                    msg = f"Invalid characters or traversal sequences in .env file content: {val}"
                    raise ValueError(msg)

    return resolved_env


class DistillationConfig(BaseModel):
    """Phase 1: Zero-Shot Distillation configuration."""

    model_config = ConfigDict(extra="forbid")
    enable: bool = Field(default=True, description="Enable distillation phase")
    mace_model_path: str = Field(
        default="mace-mp-0-medium", description="Path or name of the MACE foundation model"
    )
    uncertainty_threshold: float = Field(default=0.05, description="MACE confidence threshold")
    sampling_structures_per_system: int = Field(
        default=1000, description="Number of structures to sample per system"
    )
    device: str = Field(default="cpu", description="Device to run MACE on (e.g., cpu, cuda)")
    default_dtype: str = Field(
        default="float32", description="Default dtype for MACE (e.g., float32, float64)"
    )
    dispersion: bool = Field(default=False, description="Enable dispersion correction in MACE")
    temp_dir: str = Field(
        default="/tmp/pyacemaker_distillation",  # noqa: S108
        description="Path to temporary directory for distillation"
    )
    output_dir: str = Field(
        default="./active_learning/distillation_outputs", description="Path to save distillation outputs"
    )
    model_storage_path: str = Field(
        default="./potentials/mace_models", description="Path to cache MACE foundation models"
    )

    @model_validator(mode="after")
    def validate_thresholds_and_samples(self) -> "DistillationConfig":
        valid_foundational_names = ["mace-mp-0-medium", "mace-mp-0-large", "mace-mp-0-small"]

        # Check if it's a known model name
        if self.mace_model_path not in valid_foundational_names:
            # If it's a path, validate it
            if ".." in self.mace_model_path:
                msg = f"Path traversal sequences (..) are not allowed in mace_model_path: {self.mace_model_path}"
                raise ValueError(msg)
            # Extension check
            if not self.mace_model_path.endswith(".model") and not self.mace_model_path.endswith(
                ".pt"
            ):
                msg = f"Unknown model name or unsupported extension in mace_model_path: {self.mace_model_path}. Must be one of {valid_foundational_names} or end in .pt/.model"
                raise ValueError(msg)

        if self.uncertainty_threshold <= 0.0:
            msg = "uncertainty_threshold must be strictly positive"
            raise ValueError(msg)
        if self.sampling_structures_per_system <= 0:
            msg = "sampling_structures_per_system must be an integer strictly greater than zero"
            raise ValueError(msg)
        return self


class CutoutConfig(BaseModel):
    """Phase 3: Intelligent cutout and passivation configuration."""

    model_config = ConfigDict(extra="forbid")
    core_radius: float = Field(default=3.0, description="Radius for core atoms (force weight 1.0)")
    buffer_radius: float = Field(
        default=4.0, description="Radius for buffer layer (force weight 0.0)"
    )
    enable_pre_relaxation: bool = Field(
        default=True, description="Enable MACE pre-relaxation of buffer"
    )
    enable_passivation: bool = Field(
        default=True, description="Enable automatic surface passivation"
    )
    passivation_element: str = Field(default="H", description="Element used for passivation")

    @model_validator(mode="after")
    def validate_radii(self) -> "CutoutConfig":
        if self.core_radius <= 0.0:
            msg = "core_radius must be strictly positive"
            raise ValueError(msg)
        if self.buffer_radius <= 0.0:
            msg = "buffer_radius must be strictly positive"
            raise ValueError(msg)
        if self.buffer_radius <= self.core_radius:
            msg = (
                f"Buffer radius ({self.buffer_radius}) must be strictly greater than "
                f"core radius ({self.core_radius}) to ensure a valid physical buffer zone exists."
            )
            raise ValueError(msg)
        return self


class LoopStrategyConfig(BaseModel):
    """High-level Active Learning loop strategy configuration."""

    model_config = ConfigDict(extra="forbid")
    use_tiered_oracle: bool = Field(default=True, description="Enable tiered oracle logic")
    incremental_update: bool = Field(default=True, description="Enable incremental delta learning")
    replay_buffer_size: int = Field(
        default=500, description="Size of history replay buffer to prevent catastrophic forgetting"
    )
    baseline_potential_type: str = Field(
        default="LJ", description="Baseline physical potential type (e.g., LJ)"
    )
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)
    checkpoint_interval: int = Field(
        default=5, description="Frequency (in iterations) to execute full hard-checkpoints"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of orchestration loop retries on transient failures"
    )
    timeout_seconds: int = Field(
        default=86400, description="Maximum wall-clock timeout in seconds for a complete loop iteration"
    )

    @model_validator(mode="after")
    def validate_strategy_consistency(self) -> "LoopStrategyConfig":
        if self.incremental_update and not self.use_tiered_oracle:
            msg = "incremental_update cannot be True when use_tiered_oracle is False"
            raise ValueError(msg)
        return self


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
    def validate_env_content(cls, values: dict[str, Any]) -> dict[str, Any]:
        expected_base = Path.cwd().resolve(strict=True)
        env_file = expected_base / ".env"

        if env_file.exists():
            resolved_env = _validate_env_file_security(env_file, expected_base)

            from dotenv import dotenv_values

            # Using well-tested python-dotenv for parsing
            env_vars = dotenv_values(resolved_env)

            for key, val in env_vars.items():
                if val is None:
                    continue
                _validate_env_key(key)
                _validate_env_value(val)

                # We inject the parsed values directly into the configuration dict
                clean_key = key.replace("MLIP_", "").lower()
                if val.lower() in ("true", "1", "yes"):
                    values[clean_key] = True
                elif val.lower() in ("false", "0", "no"):
                    values[clean_key] = False
                else:
                    try:
                        if "." in val:
                            values[clean_key] = float(val)
                        else:
                            values[clean_key] = int(val)
                    except ValueError:
                        values[clean_key] = val

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
    distillation_config: DistillationConfig = Field(default_factory=DistillationConfig)
    cutout_config: CutoutConfig = Field(default_factory=CutoutConfig)
    loop_strategy: LoopStrategyConfig = Field(default_factory=LoopStrategyConfig)

    @field_validator("project_root")
    @classmethod
    def validate_project_root(cls, v: Path) -> Path:
        if ".." in str(v):
            msg = "Path traversal sequences (..) are not allowed in project_root"
            raise ValueError(msg)
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
