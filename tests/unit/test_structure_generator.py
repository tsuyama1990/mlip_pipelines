from ase import Atoms

from src.domain_models.config import InterfaceTarget, StructureGeneratorConfig
from src.generators.structure_generator import StructureGenerator


def test_generate_interface() -> None:
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)
    target = InterfaceTarget(element1="FePt", element2="MgO", face1="Fe", face2="Mg")

    interface = generator.generate_interface(target)

    assert isinstance(interface, Atoms)
    assert len(interface) > 0
    symbols = interface.get_chemical_symbols()
    assert "Fe" in symbols or "Pt" in symbols
    assert "Mg" in symbols or "O" in symbols
