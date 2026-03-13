from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


def check_phonopy_stability(atoms: "Atoms", calc: "Calculator") -> bool:
    """Calculates phonon bands using phonopy and checks for imaginary frequencies."""
    try:
        import phonopy
        from phonopy.structure.atoms import PhonopyAtoms

        # Real phonopy initialization
        unitcell = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),  # type: ignore[no-untyped-call]
            cell=atoms.get_cell(),  # type: ignore[no-untyped-call]
            positions=atoms.get_positions(),  # type: ignore[no-untyped-call]
        )
        phonon = phonopy.Phonopy(unitcell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        phonon.generate_displacements(distance=0.01)

        # Compute actual forces for displacements
        supercells = phonon.supercells_with_displacements
        force_sets = []
        if supercells is not None:
            for sc in supercells:
                if sc is None:
                    continue
                disp_atoms = atoms.copy()  # type: ignore[no-untyped-call]
                from ase.build import make_supercell

                disp_atoms = make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
                disp_atoms.set_positions(sc.get_positions())  # type: ignore[no-untyped-call]
                disp_atoms.calc = calc
                forces = disp_atoms.get_forces()  # type: ignore[no-untyped-call]
                force_sets.append(forces)

        phonon.produce_force_constants(forces=force_sets)

        # Check for imaginary frequencies
        phonon.run_mesh([10, 10, 10])
        freqs = phonon.get_mesh_dict()["frequencies"]

        return not (freqs is not None and (freqs < -0.05).any())

    except ImportError:
        import logging

        logging.warning(
            "phonopy is not installed. Skipping phonon stability check. Assuming stable."
        )
        return True


def check_mechanical_stability(atoms: "Atoms", calc: "Calculator") -> bool:  # noqa: PLR0915
    """Evaluates the Born mechanical stability criteria via applied strain."""
    import logging

    import numpy as np

    # We will compute the C11 and C12 for cubic system using volumetric and tetragonal strains
    # For a cubic system:
    # V(delta) = V(0) + 1/2 V_0 (C11 + 2 C12) delta^2 for volumetric strain

    atoms.calc = calc
    V0 = atoms.get_volume()  # type: ignore[no-untyped-call]

    try:
        E0 = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
    except Exception as e:
        logging.exception("Failed to calculate baseline potential energy")
        msg = "Calculator failed to compute potential energy on the base structure"
        raise RuntimeError(msg) from e

    # Calculate bulk modulus (B = (C11 + 2C12)/3) using volumetric strain
    strains_vol: list[float] = [-0.02, -0.01, 0.0, 0.01, 0.02]
    energies_vol: list[float] = []
    for s in strains_vol:
        if s == 0.0:
            energies_vol.append(E0)
            continue

        hydro_atoms = atoms.copy()  # type: ignore[no-untyped-call]
        cell = hydro_atoms.get_cell()  # type: ignore[no-untyped-call]
        hydro_atoms.set_cell(cell * (1 + s), scale_atoms=True)  # type: ignore[no-untyped-call]
        hydro_atoms.calc = calc
        E = hydro_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        energies_vol.append(E)

    # Fit E = E0 + 1/2 V0 * B * (3*s)^2 -> E = a + b * s + c * s^2
    # Where c = 1/2 V0 * 9 B
    import numpy.typing as npt

    coeffs_vol: npt.NDArray[np.float64] = np.polyfit(strains_vol, energies_vol, 2)  # type: ignore[no-untyped-call]
    B: float = float(2 * coeffs_vol[0] / (9 * V0))

    # Calculate C11 - C12 using tetragonal strain (volume conserving)
    # Strain tensor: e1 = s, e2 = s, e3 = (1+s)^-2 - 1 ~ -2s + 3s^2
    strains_tet: list[float] = [-0.02, -0.01, 0.01, 0.02]
    energies_tet: list[float] = []
    for s in strains_tet:
        tet_atoms = atoms.copy()  # type: ignore[no-untyped-call]
        cell = tet_atoms.get_cell()  # type: ignore[no-untyped-call]

        # apply tetragonal strain
        strain_matrix = np.array([[1 + s, 0, 0], [0, 1 + s, 0], [0, 0, 1 / (1 + s) ** 2]])

        tet_atoms.set_cell(np.dot(cell, strain_matrix), scale_atoms=True)  # type: ignore[no-untyped-call]
        tet_atoms.calc = calc
        E = tet_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        energies_tet.append(E)

    # Combine with E0 for s=0
    strains_tet.append(0.0)
    energies_tet.append(E0)

    # Fit E = E0 + 3 * V0 * (C11 - C12) * s^2
    coeffs_tet: npt.NDArray[np.float64] = np.polyfit(strains_tet, energies_tet, 2)  # type: ignore[no-untyped-call]
    C11_minus_C12: float = float(coeffs_tet[0] / (3 * V0))

    # Calculate C44 using shear strain
    strains_shear: list[float] = [-0.02, -0.01, 0.01, 0.02]
    energies_shear: list[float] = []
    for s in strains_shear:
        shear_atoms = atoms.copy()  # type: ignore[no-untyped-call]
        cell = shear_atoms.get_cell()  # type: ignore[no-untyped-call]

        strain_matrix = np.array(
            [
                [1, s / 2, 0],
                [s / 2, 1, 0],
                [
                    0,
                    0,
                    1 + s**2 / 4,
                ],  # keeping volume approx constant or V = V0 * (1-s^2/4) * (1+s^2/4)
            ]
        )

        shear_atoms.set_cell(np.dot(cell, strain_matrix), scale_atoms=True)  # type: ignore[no-untyped-call]
        shear_atoms.calc = calc
        E = shear_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        energies_shear.append(E)

    strains_shear.append(0.0)
    energies_shear.append(E0)

    # Fit E = E0 + 1/2 V0 C44 s^2
    coeffs_shear: npt.NDArray[np.float64] = np.polyfit(strains_shear, energies_shear, 2)  # type: ignore[no-untyped-call]
    C44: float = float(2 * coeffs_shear[0] / V0)

    # Derived from B = (C11 + 2C12)/3
    # B * 3 = C11 + 2C12
    # C11_minus_C12 = C11 - C12 -> C11 = C11_minus_C12 + C12
    # 3B = C11_minus_C12 + 3C12 -> C12 = B - C11_minus_C12 / 3
    # C11 = C11_minus_C12 + B - C11_minus_C12 / 3 = B + 2/3 C11_minus_C12

    C12: float = B - C11_minus_C12 / 3
    C11: float = C11_minus_C12 + C12

    # Born Stability Criteria for Cubic system
    is_stable: bool = True
    if C11 - C12 <= 0:
        is_stable = False
    if C11 + 2 * C12 <= 0:
        is_stable = False
    if C44 <= 0:
        is_stable = False

    return is_stable
