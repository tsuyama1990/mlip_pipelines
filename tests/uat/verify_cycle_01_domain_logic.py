import os
import sys

import marimo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

__generated_with = "0.1.2"
app = marimo.App()


@app.cell
def _() -> tuple:
    import os
    import shutil
    import stat
    import tempfile
    from pathlib import Path

    import marimo as mo
    import pytest
    from pydantic import ValidationError

    return Path, ValidationError, mo, os, pytest, shutil, stat, tempfile


@app.cell
def _(ValidationError, mo) -> tuple:
    mo.md("# UAT-C01-01: Validation of Active Learning Threshold Constraints and Logic")
    from src.domain_models.config import ActiveLearningThresholds

    try:
        ActiveLearningThresholds(threshold_call_dft=0.01, threshold_add_train=0.05)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert "must be strictly greater than or equal to local training addition threshold" in str(
            e
        )

    return (ActiveLearningThresholds,)


@app.cell
def _(ValidationError, mo) -> tuple:
    mo.md("# UAT-C01-02: Validation of Cluster Cutout Radii Constraints and Geometric Logic")
    from src.domain_models.config import CutoutConfig

    try:
        CutoutConfig(core_radius=6.0, buffer_radius=4.0)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert "must be strictly greater than core radius" in str(e)

    return (CutoutConfig,)


@app.cell
def _(Path, mo, tempfile, os, shutil) -> tuple:
    mo.md("# UAT-C01-03: Legacy Configuration Backward Compatibility and Safe Defaults")
    from src.domain_models.config import (
        DynamicsConfig,
        OracleConfig,
        ProjectConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True) / "myproj_legacy_uat"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "README.md").touch()

    # Mock shutil.which and os.access locally since pytest monkeypatch is not strictly active during normal execution.
    import unittest.mock

    with unittest.mock.patch(
        "shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"
    ):
        with unittest.mock.patch("os.access", return_value=True):
            config = ProjectConfig(
                project_root=tmp_dir,
                system=SystemConfig(elements=["Fe", "O"]),
                dynamics=DynamicsConfig(project_root=str(tmp_dir), trusted_directories=[]),
                oracle=OracleConfig(),
                trainer=TrainerConfig(trusted_directories=[]),
                validator=ValidatorConfig(),
            )

    assert config.cutout_config.core_radius == 3.0
    assert config.cutout_config.buffer_radius == 4.0
    assert config.cutout_config.enable_passivation is True
    assert config.distillation_config.enable is True

    return (
        DynamicsConfig,
        OracleConfig,
        ProjectConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
        config,
        tmp_dir,
    )


@app.cell
def _(ValidationError, mo) -> tuple:
    mo.md("# UAT-C01-04: Validation of the Distillation Configuration Overrides")
    from src.domain_models.config import DistillationConfig

    config_dist = DistillationConfig(
        mace_model_path="my-custom-model.pt", sampling_structures_per_system=5000
    )

    assert config_dist.mace_model_path == "my-custom-model.pt"
    assert config_dist.sampling_structures_per_system == 5000

    try:
        DistillationConfig(sampling_structures_per_system=-100)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert "must be an integer strictly greater than zero" in str(e)

    return DistillationConfig, config_dist


@app.cell
def _(ActiveLearningThresholds, ValidationError, mo) -> tuple:
    mo.md("# UAT-C01-05: Handling of Unexpected Extra Fields in Configuration")

    try:
        ActiveLearningThresholds(invalid_threshold_parameter=0.05)  # type: ignore[call-arg]
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert "Extra inputs are not permitted" in str(e)

    return


if __name__ == "__main__":
    app.run()
