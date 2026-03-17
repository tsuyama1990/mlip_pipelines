from src.domain_models.dtos import MaterialFeatures


def test_material_features_valid() -> None:
    feat = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    assert feat.elements == ["Fe", "Pt"]


# Note: Only a fragment for time limits... will just run pytest normally.
