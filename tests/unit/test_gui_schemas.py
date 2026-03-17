import pytest
from pydantic import ValidationError

from src.domain_models.gui_schemas import WorkflowIntentConfig


def test_workflow_intent_tradeoff_bounds():
    with pytest.raises(ValidationError):
        WorkflowIntentConfig(target_material="PtNi", accuracy_speed_tradeoff=0)

    with pytest.raises(ValidationError):
        WorkflowIntentConfig(target_material="PtNi", accuracy_speed_tradeoff=11)

    config = WorkflowIntentConfig(target_material="PtNi", accuracy_speed_tradeoff=5)
    assert config.accuracy_speed_tradeoff == 5

def test_target_material_security():
    with pytest.raises(ValidationError):
        WorkflowIntentConfig(target_material="../etc/passwd", accuracy_speed_tradeoff=5)
