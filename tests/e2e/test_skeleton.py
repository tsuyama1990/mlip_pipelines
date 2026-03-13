from pathlib import Path
from src.domain_models.config import PipelineConfig
from src.core.orchestrator import ActiveLearningOrchestrator


def test_pipeline_skeleton(tmp_path: Path) -> None:
    """Verifies that the orchestrator goes through the halt and heal cycle."""
    config = PipelineConfig()

    orchestrator = ActiveLearningOrchestrator(config)

    # Run one cycle
    result = orchestrator.run_cycle()

    # Can be CONVERGED or CONTINUE. It should not be ERROR or VALIDATION_FAILED.
    assert result in ["CONVERGED", "CONTINUE"]

    # If it continued, it should have created the outputs.
    if result == "CONTINUE":
        assert (Path("potentials") / "generation_001.yace").exists()
        assert (Path("active_learning") / "iter_001").exists()
