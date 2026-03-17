from src.domain_models.dtos import MaterialFeatures


def test_material_features_valid() -> None:
    feat = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    assert feat.elements == ["Fe", "Pt"]


# Note: Only a fragment for time limits... will just run pytest normally.


def test_gui_state_config_valid():
    from src.domain_models.dtos import GUIStateConfig

    config = GUIStateConfig(
        nodes=[{"id": "1"}],
        edges=[{"source": "1", "target": "2"}],
        zoom=1.5,
        viewport={"x": 0, "y": 0},
    )
    assert len(config.nodes) == 1
    assert config.zoom == 1.5


def test_gui_state_config_size_limit():
    import pytest
    from pydantic import ValidationError

    from src.domain_models.dtos import GUIStateConfig

    with pytest.raises(ValidationError, match="Too many nodes"):
        GUIStateConfig(nodes=[{"id": str(i)} for i in range(1001)])


def test_workflow_intent_config_valid():
    from src.domain_models.dtos import WorkflowIntentConfig

    config = WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=5)
    assert config.target_material == "Pt-Ni"
    assert config.accuracy_speed_tradeoff == 5


def test_workflow_intent_config_bounds():
    import pytest
    from pydantic import ValidationError

    from src.domain_models.dtos import WorkflowIntentConfig

    with pytest.raises(ValidationError):
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=0)

    with pytest.raises(ValidationError):
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=11)


def test_workflow_intent_config_security():
    import pytest
    from pydantic import ValidationError

    from src.domain_models.dtos import WorkflowIntentConfig

    with pytest.raises(ValidationError, match="Path traversal characters are not allowed"):
        WorkflowIntentConfig(target_material="../../etc/passwd", accuracy_speed_tradeoff=5)
