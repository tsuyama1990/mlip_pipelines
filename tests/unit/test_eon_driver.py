import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from src.dynamics.eon_driver import (
    _raise_empty_stdin,
    main,
    print_forces,
    read_coordinates_from_stdin,
    write_bad_structure,
)


def test_raise_empty_stdin():
    with pytest.raises(ValueError, match="Empty stdin"):
        _raise_empty_stdin()


def test_read_coordinates_from_stdin_empty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "readlines", list)
    atoms = read_coordinates_from_stdin("Fe", 5.0)
    assert atoms.get_chemical_symbols() == ["Fe"]
    assert np.allclose(atoms.get_cell(), [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])


def test_read_coordinates_from_stdin_valid(monkeypatch):
    valid_extxyz = """2
Properties=species:S:1:pos:R:3
Fe 0.0 0.0 0.0
Fe 1.0 1.0 1.0
"""
    monkeypatch.setattr(sys.stdin, "readlines", lambda: [valid_extxyz])
    atoms = read_coordinates_from_stdin("Fe", 5.0)
    assert len(atoms) == 2


def test_read_coordinates_from_stdin_invalid(monkeypatch):
    invalid_extxyz = "invalid content"
    monkeypatch.setattr(sys.stdin, "readlines", lambda: [invalid_extxyz])
    atoms = read_coordinates_from_stdin("Fe", 5.0)
    assert atoms.get_chemical_symbols() == ["Fe"]


def test_write_bad_structure_invalid_path():
    with pytest.raises(SystemExit) as e:
        write_bad_structure("../bad.cfg", Atoms("Fe"))
    assert e.value.code == 100


def test_write_bad_structure_valid(tmp_path, monkeypatch):
    import os

    original_cwd = str(Path.cwd())
    os.chdir(tmp_path)
    try:
        write_bad_structure("bad_structure.cfg", Atoms("Fe"))
        assert (tmp_path / "bad_structure.cfg").exists()
    finally:
        os.chdir(original_cwd)


def test_print_forces(capsys):
    forces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print_forces(forces)
    captured = capsys.readouterr()
    assert captured.out == "1.0 2.0 3.0\n4.0 5.0 6.0\n"


@patch("argparse.ArgumentParser.parse_args")
def test_main_invalid_threshold(mock_args):
    mock_args.return_value = argparse.Namespace(
        threshold="invalid", potential="None", default_element="Fe", default_cell=5.0
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 100


@patch("argparse.ArgumentParser.parse_args")
def test_main_invalid_potential(mock_args):
    mock_args.return_value = argparse.Namespace(
        threshold=0.1, potential="../invalid", default_element="Fe", default_cell=5.0
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 100


@patch("argparse.ArgumentParser.parse_args")
def test_main_invalid_element(mock_args):
    mock_args.return_value = argparse.Namespace(
        threshold=0.1, potential="None", default_element="Fe1", default_cell=5.0
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 100


@patch("argparse.ArgumentParser.parse_args")
def test_main_no_potential(mock_args, monkeypatch):
    mock_args.return_value = argparse.Namespace(
        threshold=0.1, potential="None", default_element="Fe", default_cell=5.0
    )
    monkeypatch.setattr(sys.stdin, "readlines", list)
    # Even with "None", the code attempts to pass. We mock it globally if missing.
    import importlib

    if not importlib.util.find_spec("pyacemaker"):
        monkeypatch.setitem(sys.modules, "pyacemaker", MagicMock())
        monkeypatch.setitem(sys.modules, "pyacemaker.calculator", MagicMock())

    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0


@patch("argparse.ArgumentParser.parse_args")
def test_main_with_potential(mock_args, monkeypatch, capsys):
    mock_args.return_value = argparse.Namespace(
        threshold=0.1, potential="dummy.yace", default_element="Fe", default_cell=5.0
    )
    monkeypatch.setattr(sys.stdin, "readlines", list)

    # Mock pyacemaker.calculator module fully
    mock_calculator = MagicMock()
    mock_pyacemaker_cls = MagicMock(return_value=mock_calculator)
    mock_calculator_mod = MagicMock()
    mock_calculator_mod.pyacemaker = mock_pyacemaker_cls
    monkeypatch.setitem(sys.modules, "pyacemaker", MagicMock())
    monkeypatch.setitem(sys.modules, "pyacemaker.calculator", mock_calculator_mod)

    # Mock Atoms behavior
    with (
        patch("ase.Atoms.get_potential_energy", return_value=1.5),
        patch("ase.Atoms.get_forces", return_value=np.array([[0.1, 0.2, 0.3]])),
    ):
        main()

    captured = capsys.readouterr()
    assert "1.5\n0.1 0.2 0.3\n" in captured.out


@patch("argparse.ArgumentParser.parse_args")
def test_main_calc_failure(mock_args, monkeypatch, tmp_path):
    mock_args.return_value = argparse.Namespace(
        threshold=0.1, potential="dummy.yace", default_element="Fe", default_cell=5.0
    )
    monkeypatch.setattr(sys.stdin, "readlines", list)

    # Mock pyacemaker.calculator module fully
    mock_calculator = MagicMock()
    mock_pyacemaker_cls = MagicMock(return_value=mock_calculator)
    mock_calculator_mod = MagicMock()
    mock_calculator_mod.pyacemaker = mock_pyacemaker_cls
    monkeypatch.setitem(sys.modules, "pyacemaker", MagicMock())
    monkeypatch.setitem(sys.modules, "pyacemaker.calculator", mock_calculator_mod)

    import os

    original_cwd = str(Path.cwd())
    os.chdir(tmp_path)
    try:
        with patch("ase.Atoms.get_potential_energy", side_effect=Exception("Calc failed")):
            with pytest.raises(SystemExit) as e:
                main()
            assert e.value.code == 100
            assert (tmp_path / "bad_structure.cfg").exists()
    finally:
        os.chdir(original_cwd)
