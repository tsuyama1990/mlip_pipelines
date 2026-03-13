from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk

from src.domain_models.config import DFTConfig
from src.oracles.dft_oracle import DFTOracle


def test_dft_oracle_apply_periodic_embedding() -> None:
    config = DFTConfig()
    oracle = DFTOracle(config)

    atoms = bulk("Fe", cubic=True)
    orig_cell = atoms.cell

    embedded = oracle._apply_periodic_embedding(atoms)
    new_cell = embedded.cell

    # Since we strictly augment the cell by buffer_zone
    assert np.allclose(new_cell.diagonal(), orig_cell.diagonal() + 5.0)
    # the atoms shouldn't be altered in our mock implementation other than the cell
    assert len(embedded) == len(atoms)


def test_dft_oracle_apply_periodic_embedding_no_cell() -> None:
    from ase import Atoms

    config = DFTConfig()
    oracle = DFTOracle(config)

    atoms = Atoms("Fe")
    embedded = oracle._apply_periodic_embedding(atoms)
    new_cell = embedded.cell

    assert not np.allclose(new_cell.diagonal(), 0.0)


def test_dft_oracle_get_pseudos() -> None:
    config = DFTConfig()
    oracle = DFTOracle(config)
    atoms = bulk("Fe", cubic=True)

    pseudos = oracle._get_pseudos(atoms)
    assert "Fe" in pseudos
    assert pseudos["Fe"] == "Fe.upf"


@patch("shutil.which", return_value="pw.x")
def test_dft_oracle_compute_batch(mock_which: MagicMock, tmp_path: Path) -> None:
    config = DFTConfig()
    oracle = DFTOracle(config)

    atoms1 = bulk("Fe", cubic=True)
    atoms2 = bulk("Pt", cubic=True)

    with (
        patch("ase.calculators.espresso.Espresso.get_potential_energy"),
        patch("ase.calculators.espresso.Espresso.get_forces", return_value=np.zeros((2, 3))),
        patch("src.oracles.dft_oracle.Espresso") as MockEspresso,
    ):
        # Mock ASE Espresso since it fails with BadConfiguration locally
        mock_calc = MagicMock()
        MockEspresso.return_value = mock_calc
        with (
            patch("ase.Atoms.get_potential_energy", return_value=-100.0),
            patch("ase.Atoms.get_forces", return_value=np.zeros((2, 3))),
        ):
            results = oracle.compute_batch([atoms1, atoms2], tmp_path)

            assert len(results) == 2
            # Notice that in the new scalable version, the calculator is detached immediately after
            # computing the forces to save memory! So `atoms.calc is None` is now the expected behavior.
            for atoms in results:
                assert atoms.calc is None


@patch("shutil.which", return_value="pw.x")
def test_dft_oracle_compute_batch_exception_handling(mock_which: MagicMock, tmp_path: Path) -> None:
    config = DFTConfig()
    oracle = DFTOracle(config)

    atoms = bulk("Fe", cubic=True)

    with patch("src.oracles.dft_oracle.Espresso") as MockEspresso:
        mock_calc = MagicMock()
        MockEspresso.return_value = mock_calc

        # We need to mock get_potential_energy on the atoms object since dft_oracle assigns calc to atoms
        # then calls atoms.get_potential_energy()

        with (
            patch(
                "ase.Atoms.get_potential_energy", side_effect=[Exception("mocked error"), -100.0]
            ),
            patch("ase.Atoms.get_forces", return_value=np.zeros((2, 3))),
        ):
            results = oracle.compute_batch([atoms], tmp_path)

            assert len(results) == 1


@patch("shutil.which", return_value=None)
def test_dft_oracle_compute_batch_no_pw(mock_which: MagicMock, tmp_path: Path) -> None:
    config = DFTConfig()
    oracle = DFTOracle(config)

    atoms = bulk("Fe", cubic=True)

    with patch("src.oracles.dft_oracle.Espresso"):
        results = oracle.compute_batch([atoms], tmp_path)

        assert len(results) == 0
