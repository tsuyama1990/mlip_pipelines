import json
import os
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any
from ase import Atoms
from loguru import logger
from src.core.sssp import load_sssp_db, validate_pseudopotentials

class AtomicEnergyManager:
    """
    Manages isolated atomic energies (E0) for Delta Learning and referencing.
    """
    def __init__(self, pseudo_dir: str, sssp_json_path: str, calculator_factory: Callable[[str], Any]):
        """
        Initialize the AtomicEnergyManager.

        Parameters
        ----------
        pseudo_dir : str
            Directory containing pseudopotentials and where E0 cache will be stored.
        sssp_json_path : str
            Path to the SSSP JSON file.
        calculator_factory : Callable[[str], Any]
            Function that accepts an element symbol and returns an ASE calculator.
        """
        self.pseudo_dir = Path(pseudo_dir)
        self.sssp_json_path = Path(sssp_json_path)
        self.calculator_factory = calculator_factory

        # Load and validate SSSP availability (light check)
        self.sssp_db = load_sssp_db(str(self.sssp_json_path))

    def get_atomic_energies(self, elements: List[str]) -> Dict[str, float]:
        """
        Get E0 for a list of elements. Computes or loads from cache.

        Parameters
        ----------
        elements : List[str]
            List of chemical symbols.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping element -> energy.
        """
        # Validate PPs exist
        validate_pseudopotentials(str(self.pseudo_dir), elements, self.sssp_db)

        energies = {}
        unique_elements = sorted(list(set(elements)))

        for elem in unique_elements:
            cache_file = self.pseudo_dir / f"{elem}.json"

            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        energies[elem] = data["energy"]
                        logger.debug(f"Loaded E0 for {elem} from cache.")
                except Exception as e:
                    logger.warning(f"Failed to load cache for {elem}, recomputing: {e}")
                    energies[elem] = self._compute_energy(elem, cache_file)
            else:
                energies[elem] = self._compute_energy(elem, cache_file)

        return energies

    def _compute_energy(self, element: str, cache_file: Path) -> float:
        """
        Compute energy for a single isolated atom and save to cache.

        Parameters
        ----------
        element : str
            Chemical symbol.
        cache_file : Path
            Path to save the result.

        Returns
        -------
        float
            The computed energy (E0).
        """
        logger.info(f"Computing E0 for {element}...")

        # Create isolated atom
        # Box size should be large enough to avoid interaction, e.g., 10-15 A vacuum
        atoms = Atoms(element, positions=[[0, 0, 0]], cell=[15, 15, 15], pbc=True)
        atoms.center()

        # Attach calculator
        calc = self.calculator_factory(element)
        atoms.calc = calc

        try:
            energy = atoms.get_potential_energy()
        except Exception as e:
            logger.error(f"Failed to compute energy for {element}: {e}")
            raise

        # Save to cache
        data = {
            "element": element,
            "energy": energy,
            "calculator_info": str(calc) # Metadata
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Computed and cached E0 for {element}: {energy:.4f} eV")
        return energy
