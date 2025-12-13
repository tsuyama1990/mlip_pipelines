import pytest
from pydantic import ValidationError
from src.config.settings import Settings, RelaxationSettings, MACESettings

def test_default_settings():
    """Test that default settings are valid."""
    settings = Settings()
    # Assuming device validation logic allows 'cpu' fallback or 'cuda'
    assert settings.mace.device in ["cpu", "cuda"]
    assert settings.relax.fmax > 0
    assert settings.relax.optimizer == "LBFGS"

def test_validation_error():
    """Test validation of invalid values."""
    # Test invalid device
    with pytest.raises(ValidationError):
        MACESettings(device="invalid_device")

    # Test invalid fmax (should be > 0)
    with pytest.raises(ValidationError):
        RelaxationSettings(fmax=-0.1)

def test_custom_settings():
    """Test overriding defaults."""
    relax = RelaxationSettings(steps=500, fmax=0.001)
    assert relax.steps == 500
    assert relax.fmax == 0.001
