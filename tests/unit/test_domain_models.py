import pytest

from src.domain_models.dtos import GUIStateConfig, MaterialFeatures, WorkflowIntentConfig


def test_material_features_valid() -> None:
    feat = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    assert feat.elements == ["Fe", "Pt"]


# Note: Only a fragment for time limits... will just run pytest normally.


def test_workflow_intent_config_valid():
    intent = WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=5)
    assert intent.target_material == "Pt-Ni"
    assert intent.accuracy_speed_tradeoff == 5

def test_workflow_intent_config_invalid_tradeoff():
    with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=0)

    with pytest.raises(ValueError, match="Input should be less than or equal to 10"):
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=11)

def test_workflow_intent_config_invalid_material():
    with pytest.raises(ValueError, match="Path traversal sequences"):
        WorkflowIntentConfig(target_material="../Pt-Ni", accuracy_speed_tradeoff=5)

    with pytest.raises(ValueError, match="Forbidden shell character"):
        WorkflowIntentConfig(target_material="Pt-Ni; rm -rf /", accuracy_speed_tradeoff=5)

def test_gui_state_config():
    state = GUIStateConfig(react_flow_state={"nodes": [{"id": "1"}], "edges": []})
    assert len(state.react_flow_state["nodes"]) == 1

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        GUIStateConfig(react_flow_state={}, unknown_field=123)  # type: ignore[call-arg]
