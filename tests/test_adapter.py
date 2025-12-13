import pytest
from ase import Atoms
from src.core.generators.adapter import ExternalGeneratorAdapter

def test_adapter_initialization(default_settings):
    """
    Test that the adapter initializes the external generator correctly.
    """
    adapter = ExternalGeneratorAdapter(default_settings.generator)
    assert adapter.generator is not None
    # Verify it has the right config type (duck typing check)
    assert hasattr(adapter.generator, "config")
    assert adapter.generator.config.type == "alloy"

def test_adapter_generation(default_settings):
    """
    Test structure generation via adapter.
    """
    # Use a simple target to avoid complex external logic issues
    default_settings.generator.target_element = "Cu" # Cu is safe default for Alloy
    # Must clear elements list if it was set in fixture, otherwise pydantic validator might prefer existing list
    # However, in conftest default_settings: generator=GeneratorSettings(target_element="Si", supercell_size=1, vacancy_concentration=0.0)
    # The elements field defaults to None, and validator populates it from target_element.
    # When we modify target_element AFTER instantiation, elements is already ["Si"].
    # We must reset elements to None to trigger re-validation or manually set it.
    default_settings.generator.elements = ["Cu"]

    adapter = ExternalGeneratorAdapter(default_settings.generator)

    atoms = adapter.generate()

    assert isinstance(atoms, Atoms)
    assert len(atoms) > 0
    assert "Cu" in atoms.symbols
