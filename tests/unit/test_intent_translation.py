from src.domain_models.config import ProjectConfig
from src.domain_models.gui_schemas import WorkflowIntentConfig


def test_intent_translation_max_speed(mock_project_config: ProjectConfig) -> None:
    intent = WorkflowIntentConfig(target_material="Pt", accuracy_speed_tradeoff=1)

    # We directly test the logic on the fixture without Pydantic full validation
    # because Pydantic deep validation requires external binaries that aren't mocked here
    mock_project_config.intent = intent
    mock_project_config.translate_intent_to_hyperparameters()

    # 1 -> 0.15 threshold
    assert mock_project_config.distillation_config.uncertainty_threshold == 0.15
    # 1 -> 50 replay buffer
    assert mock_project_config.loop_strategy.replay_buffer_size == 50


def test_intent_translation_max_accuracy(mock_project_config: ProjectConfig) -> None:
    intent = WorkflowIntentConfig(target_material="Pt", accuracy_speed_tradeoff=10)
    mock_project_config.intent = intent
    mock_project_config.translate_intent_to_hyperparameters()

    # 10 -> approx 0.02 threshold (0.15 - 9*0.0144 = 0.0204 -> 0.0204)
    assert mock_project_config.distillation_config.uncertainty_threshold == 0.0204
    # 10 -> 500 replay buffer (50 + 9*50)
    assert mock_project_config.loop_strategy.replay_buffer_size == 500


def test_intent_translation_median(mock_project_config: ProjectConfig) -> None:
    intent = WorkflowIntentConfig(target_material="Pt", accuracy_speed_tradeoff=5)
    mock_project_config.intent = intent
    mock_project_config.translate_intent_to_hyperparameters()

    # 5 -> (0.15 - 4*0.0144 = 0.0924)
    assert mock_project_config.distillation_config.uncertainty_threshold == 0.0924
    # 5 -> 250 replay buffer
    assert mock_project_config.loop_strategy.replay_buffer_size == 250


def test_backward_compatibility(mock_project_config: ProjectConfig) -> None:
    mock_project_config.translate_intent_to_hyperparameters()

    # DistillationConfig uncertainty_threshold default is 0.05
    assert mock_project_config.distillation_config.uncertainty_threshold == 0.05

    # LoopStrategyConfig replay_buffer_size given in fixture is 1000
    assert mock_project_config.loop_strategy.replay_buffer_size == 1000
