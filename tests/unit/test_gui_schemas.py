import pytest
from pydantic import ValidationError

from src.domain_models.gui_schemas import GUIStateConfig, WorkflowIntentConfig


def test_gui_state_config_size_limit() -> None:
    # 1MB is 1048576 bytes
    # Create a dict that will serialize to more than 1MB
    large_dict = {"data": "x" * 2000000}
    with pytest.raises(ValidationError, match="GUIStateConfig state exceeds maximum allowed size"):
        GUIStateConfig(state=large_dict)

def test_workflow_intent_config_validation_valid() -> None:
    intent = WorkflowIntentConfig(target_material="Fe-Pt", accuracy_speed_tradeoff=5, enable_auto_hpo=True)
    assert intent.target_material == "Fe-Pt"
    assert intent.accuracy_speed_tradeoff == 5
    assert intent.enable_auto_hpo is True

def test_workflow_intent_config_validation_invalid_bounds() -> None:
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
        WorkflowIntentConfig(target_material="Fe", accuracy_speed_tradeoff=0)

    with pytest.raises(ValidationError, match="Input should be less than or equal to 10"):
        WorkflowIntentConfig(target_material="Fe", accuracy_speed_tradeoff=11)

def test_workflow_intent_config_validation_path_traversal() -> None:
    with pytest.raises(ValidationError, match="Invalid characters or traversal sequences in string"):
        WorkflowIntentConfig(target_material="../../etc/passwd", accuracy_speed_tradeoff=5)
