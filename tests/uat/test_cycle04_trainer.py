from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from src.domain_models.config import TrainerConfig
from src.trainers.ace_trainer import PacemakerWrapper


def test_uat_04_01_d_optimality_active_set_filtering(tmp_path: Path) -> None:
    """
    UAT-04-01: D-Optimality Active Set Filtering

    GIVEN a list of 20 highly correlated `Atoms` candidate structures
    AND a single `Atoms` anchor structure `s0`
    AND the `PacemakerWrapper` is configured with `n=5` for active set selection
    WHEN `select_local_active_set()` is called (mocking the `pace_activeset` subprocess)
    THEN the returned list of structures should contain exactly 5 unique `Atoms` objects
    AND the anchor structure `s0` should be explicitly included in this list.
    """
    # GIVEN
    config = TrainerConfig(active_set_size=5, trusted_directories=[])
    wrapper = PacemakerWrapper(config)
    anchor = Atoms("Fe", positions=[(0, 0, 0)])
    candidates = [Atoms("Fe", positions=[(i * 0.1, 0, 0)]) for i in range(1, 21)]

    def mock_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output") + 1
        out_file = Path(cmd[out_idx].strip("'\""))
        from ase.io import write

        # Return the anchor and 4 candidates
        selected_structures = [anchor, *candidates[:4]]
        write(str(out_file), selected_structures, format="extxyz")
        return MagicMock(returncode=0)

    # WHEN
    with patch("subprocess.run", side_effect=mock_run):
        selected = wrapper.select_local_active_set(candidates, anchor=anchor, n=5)

    # THEN
    assert len(selected) == 5
    # Verify anchor is included (by position in this mock case)
    assert selected[0].positions[0][0] == 0.0
    assert selected[0].symbols == "Fe"


def test_uat_04_02_physics_informed_delta_learning_configuration(tmp_path: Path) -> None:
    """
    UAT-04-02: Physics-Informed Delta Learning configuration

    GIVEN a `TrainerConfig` specifying `baseline_potential="zbl"`
    AND a valid `dataset.extxyz` file
    AND an `initial_potential.yace` file
    WHEN `train()` is called
    THEN the constructed command list passed to `subprocess.run` (mocked) should include `--baseline_potential zbl`
    AND it should include `--initial_potential initial_potential.yace`
    AND it should include `--dataset dataset.extxyz`.
    """
    # GIVEN
    config = TrainerConfig(baseline_potential="zbl", trusted_directories=[])
    wrapper = PacemakerWrapper(config)

    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("10\ncontent")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    initial_potential = tmp_path / "initial_potential.yace"
    initial_potential.write_text("dummy")

    # WHEN
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        wrapper.train(dataset, initial_potential, out_dir)

        # THEN
        cmd_args = mock_run.call_args[0][0]

        # Checking flags
        assert "--baseline_potential" in cmd_args
        idx_baseline = cmd_args.index("--baseline_potential")
        assert cmd_args[idx_baseline + 1] == "zbl"

        assert "--initial_potential" in cmd_args
        idx_init_pot = cmd_args.index("--initial_potential")
        assert cmd_args[idx_init_pot + 1] == str(initial_potential.resolve(strict=True))

        assert "--dataset" in cmd_args
        idx_dataset = cmd_args.index("--dataset")
        assert cmd_args[idx_dataset + 1] == str(dataset.resolve(strict=True))


def test_uat_04_03_secure_subprocess_execution(tmp_path: Path) -> None:
    """
    UAT-04-03: Secure Subprocess Execution

    GIVEN a `TrainerConfig` where a parameter contains potential shell meta-characters (e.g., `baseline_potential="zbl; rm -rf /"`)
    WHEN `train()` attempts to construct and execute the `pace_train` command
    THEN the command should be passed as a strictly sanitized list of strings to `subprocess.run(shell=False)`
    AND the shell meta-characters should be treated as literal string arguments to the binary, preventing arbitrary code execution.
    """
    # We first verify that the config validation rejects this.
    # If the domain model does not reject it, the builder should.

    # GIVEN
    dataset = tmp_path / "dataset.extxyz"
    dataset.write_text("10\ncontent")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Create config bypassing validation (if any) or see if it gets rejected at runtime.
    config = TrainerConfig(trusted_directories=[])
    wrapper = PacemakerWrapper(config)
    wrapper.config.baseline_potential = "zbl; rm -rf /"

    # WHEN & THEN
    with pytest.raises(ValueError, match="Invalid baseline potential format"):
        wrapper.train(dataset, None, out_dir)
