import pytest
from pydantic import ValidationError

from src.domain_models.gui_schemas import GUIStateConfig, WorkflowIntentConfig


def test_workflow_intent_valid():
    intent = WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=5)
    assert intent.target_material == "Pt-Ni"
    assert intent.accuracy_speed_tradeoff == 5


def test_workflow_intent_invalid_tradeoff():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=0)

    with pytest.raises(ValidationError, match="less than or equal to 10"):
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=11)


def test_workflow_intent_invalid_material():
    with pytest.raises(
        ValidationError, match="Path traversal or directory characters are not allowed"
    ):
        WorkflowIntentConfig(target_material="../etc/passwd", accuracy_speed_tradeoff=5)

    with pytest.raises(ValidationError, match="Shell injection characters are not allowed"):
        WorkflowIntentConfig(target_material="Pt-Ni; rm -rf", accuracy_speed_tradeoff=5)


def test_gui_state_valid():
    state = GUIStateConfig(nodes=[{"id": "1"}], edges=[])
    assert len(state.nodes) == 1


def test_gui_state_invalid_size():
    with pytest.raises(ValidationError, match="Too many nodes"):
        GUIStateConfig(nodes=[{"id": str(i)} for i in range(1001)], edges=[])

    with pytest.raises(ValidationError, match="Too many edges"):
        GUIStateConfig(nodes=[], edges=[{"id": str(i)} for i in range(10001)])
