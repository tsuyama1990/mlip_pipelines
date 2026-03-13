
import pytest

from src.domain_models.config import MaterialConfig, PipelineConfig


@pytest.fixture
def mock_material_config() -> MaterialConfig:
    return MaterialConfig(
        elements=["Fe", "Pt"],
        atomic_numbers=[26, 78],
        masses=[55.845, 195.084],
        band_gap=0.0,
        melting_point=1500.0,
        bulk_modulus=180.0,
        crystal="bcc",
        a=2.8665
    )

@pytest.fixture
def mock_pipeline_config(mock_material_config: MaterialConfig) -> PipelineConfig:
    return PipelineConfig(material=mock_material_config)
