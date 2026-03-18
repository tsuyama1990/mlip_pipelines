import re

with open("tests/uat/verify_cycle_06_domain_logic.py", "r") as f:
    text = f.read()

# Fix DummyConfig to accept project_root parameter
replacement = '''@pytest.fixture
def DummyConfig(tmp_path: Path):
    from src.domain_models.config import (
        ProjectConfig, SystemConfig, DynamicsConfig, OracleConfig, TrainerConfig,
        ValidatorConfig, DistillationConfig, LoopStrategyConfig, ActiveLearningThresholds, CutoutConfig
    )

    def _create(project_root_str=None):
        pr = project_root_str or str(tmp_path)
        return ProjectConfig(
            project_root=pr,
            system=SystemConfig(elements=["Fe", "O"]),
            dynamics=DynamicsConfig(project_root=pr, trusted_directories=[]),
            oracle=OracleConfig(),
            trainer=TrainerConfig(trusted_directories=[]),
            validator=ValidatorConfig(),
            distillation_config=DistillationConfig(temp_dir=pr, output_dir=pr, model_storage_path=pr),
            loop_strategy=LoopStrategyConfig(replay_buffer_size=10, checkpoint_interval=5, max_retries=3, timeout_seconds=3600),
            cutout_config=CutoutConfig()
        )
    return _create'''

text = re.sub(
    r'@pytest\.fixture\ndef DummyConfig\(tmp_path: Path\):[\s\S]*?return _create',
    replacement,
    text, count=1
)

with open("tests/uat/verify_cycle_06_domain_logic.py", "w") as f:
    f.write(text)
