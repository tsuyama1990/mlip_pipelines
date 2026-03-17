import pytest
from pydantic import ValidationError

from src.domain_models.gui_schemas import GUIStateConfig, WorkflowIntentConfig


def test_workflow_intent_config_valid() -> None:
    intent = WorkflowIntentConfig(
        target_material="Pt-Ni", accuracy_speed_tradeoff=5, enable_auto_hpo=True
    )
    assert intent.target_material == "Pt-Ni"
    assert intent.accuracy_speed_tradeoff == 5
    assert intent.enable_auto_hpo is True


def test_workflow_intent_config_out_of_bounds() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=11)
    assert "accuracy_speed_tradeoff" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=0)
    assert "accuracy_speed_tradeoff" in str(exc_info.value)


def test_workflow_intent_config_security_rejection() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WorkflowIntentConfig(target_material="../../etc/passwd", accuracy_speed_tradeoff=5)
    assert "target_material" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        WorkflowIntentConfig(target_material="Pt-Ni; rm -rf /", accuracy_speed_tradeoff=5)
    assert "target_material" in str(exc_info.value)


def test_gui_state_config_valid() -> None:
    state = GUIStateConfig(flow_data={"nodes": [], "edges": []})
    assert "nodes" in state.flow_data


def test_gui_state_config_too_large() -> None:
    # Construct a dict that will exceed 1MB when dumped to JSON
    large_data = {"key": "A" * (1024 * 1024 + 10)}
    with pytest.raises(ValidationError) as exc_info:
        GUIStateConfig(flow_data=large_data)
    assert "flow_data" in str(exc_info.value)
