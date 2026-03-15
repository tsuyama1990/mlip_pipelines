import typing

import marimo

__generated_with = "0.2.1"
app = marimo.App(width="medium")

@app.cell
def setup() -> tuple:
    import os
    import sys
    from pathlib import Path

    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    import typing

    import numpy as np
    from ase import Atoms
    from ase.build import bulk
    from ase.calculators.calculator import Calculator, all_changes

    from src.domain_models.config import CutoutConfig
    from src.generators.extraction import (
        _passivate_surface,
        _pre_relax_buffer,
        extract_intelligent_cluster,
    )

    class SurrogateCalculator(Calculator):
        implemented_properties: typing.ClassVar[list[str]] = ["energy", "forces"]

        def __init__(self, **kwargs: typing.Any) -> None:
            super().__init__(**kwargs)
            self.k = 1.0
            self.call_count = 0

        def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list[str] | None = None,
            system_changes: list[str] | None = all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            self.call_count += 1
            if atoms is None:
                return

            pos = atoms.positions
            com = atoms.get_center_of_mass()
            forces = -self.k * (pos - com)
            energy = np.sum(0.5 * self.k * np.sum((pos - com) ** 2, axis=1))
            self.results["forces"] = forces
            self.results["energy"] = energy

    class FailingSurrogateCalculator(Calculator):
        implemented_properties: typing.ClassVar[list[str]] = ["energy", "forces"]

        def calculate(self, *args: typing.Any, **kwargs: typing.Any) -> None:
            msg = "MACE optimization failed!"
            raise RuntimeError(msg)

    return (
        sys,
        os,
        Atoms,
        bulk,
        Calculator,
        all_changes,
        CutoutConfig,
        extract_intelligent_cluster,
        _passivate_surface,
        _pre_relax_buffer,
        np,
        SurrogateCalculator,
        FailingSurrogateCalculator,
    )


@app.cell
def test_scenario_1(
    Atoms: typing.Any,  # noqa: N803
    bulk: typing.Any,
    extract_intelligent_cluster: typing.Any,
    CutoutConfig: typing.Any,  # noqa: N803
    np: typing.Any,
) -> tuple:
    bulk_atoms = bulk("Cu", "fcc", a=3.6).repeat((5, 5, 5))
    center_idx = 62  # some arbitrary core atom

    config = CutoutConfig(
        core_radius=3.0,
        buffer_radius=5.0,
        enable_passivation=False,
        enable_pre_relaxation=False,
        passivation_element="H",
    )

    result = extract_intelligent_cluster(bulk_atoms, center_idx, config)
    cluster = result.cluster

    assert "force_weights" in cluster.arrays, "force_weights missing"

    force_weights_1 = cluster.arrays["force_weights"]
    distances = np.linalg.norm(cluster.positions, axis=1)

    assert np.all(distances[force_weights_1 == 1.0] <= 3.0), "Core atoms exceed core radius"
    assert np.all(
        (distances[force_weights_1 == 0.0] > 3.0) & (distances[force_weights_1 == 0.0] <= 5.0)
    ), "Buffer atoms outside buffer zone"
    return (bulk_atoms, center_idx, config, result, cluster, force_weights_1, distances)


@app.cell
def test_scenario_2(
    bulk: typing.Any,
    extract_intelligent_cluster: typing.Any,
    CutoutConfig: typing.Any,  # noqa: N803
    np: typing.Any,
) -> tuple:
    bulk_atoms_mgo = bulk("MgO", "rocksalt", a=4.21).repeat((3, 3, 3))
    center_idx_mgo = 27

    config_passivate = CutoutConfig(
        core_radius=2.6,
        buffer_radius=4.5,
        enable_passivation=True,
        enable_pre_relaxation=False,
        passivation_element="H",
    )

    result_mgo = extract_intelligent_cluster(bulk_atoms_mgo, center_idx_mgo, config_passivate)
    cluster_mgo = result_mgo.cluster
    added_count = result_mgo.passivation_atoms_added

    assert added_count > 0, "No passivation atoms added"
    assert "H" in cluster_mgo.get_chemical_symbols(), "Passivating element 'H' not found in cluster"

    force_weights_2 = cluster_mgo.arrays["force_weights"]
    symbols = np.array(cluster_mgo.get_chemical_symbols())
    h_weights = force_weights_2[symbols == "H"]
    assert np.all(h_weights == 0.0), "Passivating atoms have non-zero force weight"

    return (
        bulk_atoms_mgo,
        center_idx_mgo,
        config_passivate,
        result_mgo,
        cluster_mgo,
        added_count,
        symbols,
        force_weights_2,
    )


@app.cell
def test_scenario_3(
    bulk: typing.Any,
    extract_intelligent_cluster: typing.Any,
    CutoutConfig: typing.Any,  # noqa: N803
    SurrogateCalculator: typing.Any,  # noqa: N803
    np: typing.Any,
) -> tuple:
    bulk_atoms_cu = bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3))
    center_idx_cu = 13

    config_relax = CutoutConfig(
        core_radius=2.0,
        buffer_radius=4.0,
        enable_passivation=False,
        enable_pre_relaxation=True,
        passivation_element="H",
    )

    config_norelax = config_relax.model_copy(update={"enable_pre_relaxation": False})
    unrelaxed_cluster = extract_intelligent_cluster(
        bulk_atoms_cu, center_idx_cu, config_norelax
    ).cluster

    mock_calc = SurrogateCalculator()
    result_relax = extract_intelligent_cluster(
        bulk_atoms_cu, center_idx_cu, config_relax, calculator=mock_calc
    )
    relaxed_cluster = result_relax.cluster

    assert mock_calc.call_count > 0, "Calculator was not called"

    fw_3 = relaxed_cluster.arrays["force_weights"]

    core_unrelaxed = unrelaxed_cluster.positions[fw_3 == 1.0]
    core_relaxed = relaxed_cluster.positions[fw_3 == 1.0]
    np.testing.assert_allclose(
        core_unrelaxed, core_relaxed, atol=1e-10, err_msg="Core atoms moved"
    )

    buffer_unrelaxed = unrelaxed_cluster.positions[fw_3 == 0.0]
    buffer_relaxed = relaxed_cluster.positions[fw_3 == 0.0]

    moved = not np.allclose(buffer_unrelaxed, buffer_relaxed, atol=1e-5)
    assert moved, "Buffer atoms did not move"

    return (
        bulk_atoms_cu,
        center_idx_cu,
        config_relax,
        unrelaxed_cluster,
        mock_calc,
        result_relax,
        relaxed_cluster,
        fw_3,
    )


@app.cell
def test_scenario_4(
    bulk: typing.Any,
    extract_intelligent_cluster: typing.Any,
    CutoutConfig: typing.Any,  # noqa: N803
    np: typing.Any,
) -> tuple:
    bulk_atoms_edge = bulk("Fe", "bcc", a=2.86).repeat((3, 3, 3))
    edge_idx = 0

    config_edge = CutoutConfig(
        core_radius=3.0,
        buffer_radius=5.0,
        enable_passivation=False,
        enable_pre_relaxation=False,
    )

    result_edge = extract_intelligent_cluster(bulk_atoms_edge, edge_idx, config_edge)
    cluster_edge = result_edge.cluster

    assert len(cluster_edge) > 1, "Failed to extract neighbors across PBC"

    distances_4 = np.linalg.norm(cluster_edge.positions, axis=1)
    assert np.all(distances_4 <= 5.0), "Extracted atoms have wrong positions"

    return (bulk_atoms_edge, edge_idx, config_edge, result_edge, cluster_edge, distances_4)


@app.cell
def test_scenario_5(
    bulk: typing.Any,
    extract_intelligent_cluster: typing.Any,
    CutoutConfig: typing.Any,  # noqa: N803
    FailingSurrogateCalculator: typing.Any,  # noqa: N803
) -> tuple:
    bulk_atoms_fail = bulk("Cu", "fcc", a=3.6).repeat((2, 2, 2))

    config_fail = CutoutConfig(
        core_radius=2.0,
        buffer_radius=4.0,
        enable_passivation=False,
        enable_pre_relaxation=True,
    )

    fail_calc = FailingSurrogateCalculator()

    result_fail = extract_intelligent_cluster(
        bulk_atoms_fail, 0, config_fail, calculator=fail_calc
    )

    assert result_fail.cluster is not None
    assert len(result_fail.cluster) > 0

    return (bulk_atoms_fail, config_fail, fail_calc, result_fail)


if __name__ == "__main__":
    app.run()
