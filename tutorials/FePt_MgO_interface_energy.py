import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def __1(): # type: ignore[no-untyped-def]

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    return sys, Path


@app.cell
def __2(sys, Path): # type: ignore[no-untyped-def]

    import tempfile

    from src.domain_models.config import (
        ActiveLearningThresholds,
        CutoutConfig,
        DistillationConfig,
        DynamicsConfig,
        LoopStrategyConfig,
        OracleConfig,
        ProjectConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )

    tmp_dir = Path(tempfile.mkdtemp())
    (tmp_dir / "pyproject.toml").touch()

    config = ProjectConfig(
        project_root=tmp_dir,
        system=SystemConfig(elements=["Fe", "Pt", "Mg", "O"]),
        dynamics=DynamicsConfig(project_root=str(tmp_dir), thresholds=ActiveLearningThresholds(), trusted_directories=[]),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig(),
        distillation_config=DistillationConfig(
            temp_dir=str(tmp_dir), output_dir=str(tmp_dir), model_storage_path=str(tmp_dir)
        ),
        loop_strategy=LoopStrategyConfig(
            replay_buffer_size=100, checkpoint_interval=10, max_retries=3, timeout_seconds=3600
        ),
        cutout_config=CutoutConfig(),
    )
    print("Config successfully loaded!")
    return config, tmp_dir


@app.cell
def __3(config, tmp_dir): # type: ignore[no-untyped-def]

    from src.core.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)
    print("Orchestrator successfully initialized!")
    return (orchestrator,)


if __name__ == "__main__":
    app.run()
