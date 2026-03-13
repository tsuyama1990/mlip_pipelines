import pytest
from pathlib import Path
from src.domain_models.config import PipelineConfig

def test_pipeline_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
project_name: test_project
lammps:
  temperature: 500.0
dft:
  kspacing: 0.1
training:
  max_epochs: 100
validation:
  rmse_energy_threshold: 1.0
otf_loop:
  uncertainty_threshold: 3.0
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = PipelineConfig.from_yaml(config_file)

    assert config.project_name == "test_project"
    assert config.lammps.temperature == 500.0
    assert config.dft.kspacing == 0.1
    assert config.training.max_epochs == 100
    assert config.validation.rmse_energy_threshold == 1.0
    assert config.otf_loop.uncertainty_threshold == 3.0

def test_pipeline_config_from_empty_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "empty_config.yaml"
    config_file.write_text("")

    config = PipelineConfig.from_yaml(config_file)

    assert config.project_name == "mlip_project"
    assert config.lammps.temperature == 300.0
