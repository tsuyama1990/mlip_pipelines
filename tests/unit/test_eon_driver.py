import sys
from pathlib import Path

import pytest
from ase import Atoms

from src.dynamics import eon_driver


def test_read_coordinates_empty_stdin(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr(sys.stdin, "read", lambda size=0: "")

    atoms = eon_driver.read_coordinates_from_stdin("Fe", 5.0)
    assert len(atoms) == 1
    assert atoms.get_chemical_symbols() == ["Fe"]
    assert atoms.get_cell()[0][0] == 5.0


def test_write_bad_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "bad.cfg"
    atoms = Atoms("Fe", positions=[[0, 0, 0]])
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    eon_driver.write_bad_structure(str(path), atoms)
    assert path.exists()


def test_write_bad_structure_invalid_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: Path
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    with pytest.raises(SystemExit) as e:
        eon_driver.write_bad_structure("../bad.cfg", Atoms("Fe"))
    assert e.value.code == 100
    out, err = capsys.readouterr()
    assert "Invalid path" in err


def test_read_coordinates_invalid_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):

    # return the invalid format on the first read, then empty
    calls = [0]

    def mock_read(size=-1):
        if calls[0] == 0:
            calls[0] += 1
            return "invalid format\n"
        return ""

    monkeypatch.setattr(sys.stdin, "read", mock_read)

    atoms = eon_driver.read_coordinates_from_stdin("Pt", 10.0)
    assert len(atoms) == 1
    assert atoms.get_chemical_symbols() == ["Pt"]

    out, err = capsys.readouterr()
    assert "Failed to parse input stream" in err


def test_print_forces(capsys: pytest.CaptureFixture):
    forces = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    eon_driver.print_forces(forces)
    out, err = capsys.readouterr()
    assert "1.0 2.0 3.0\n4.0 5.0 6.0\n" in out


def test_main_empty_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import sys

    monkeypatch.setattr(sys.stdin, "read", lambda size=0: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eon_driver.py",
            "--threshold",
            "5.0",
            "--potential",
            "None",
            "--default_element",
            "Fe",
            "--default_cell",
            "5.0",
        ],
    )

    # mock sys.modules to simulate pyacemaker import
    class DummyCalc:
        pass

    import sys

    class MockPyacemakerModule:
        pyacemaker = DummyCalc

    sys.modules["pyacemaker"] = MockPyacemakerModule()
    sys.modules["pyacemaker.calculator"] = MockPyacemakerModule()

    with pytest.raises(SystemExit) as e:
        eon_driver.main()
    assert e.value.code == 0


def test_main_invalid_threshold(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    import sys

    monkeypatch.setattr(sys, "argv", ["eon_driver.py", "--threshold", "invalid"])
    with pytest.raises(SystemExit) as e:
        eon_driver.main()
    assert e.value.code == 100


def test_main_invalid_potential(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    import sys

    monkeypatch.setattr(
        sys, "argv", ["eon_driver.py", "--threshold", "5.0", "--potential", "../test.yace"]
    )
    with pytest.raises(SystemExit) as e:
        eon_driver.main()
    assert e.value.code == 100
    out, err = capsys.readouterr()
    assert "Potential path contains invalid characters" in err


def test_main_invalid_element(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    import sys

    monkeypatch.setattr(
        sys, "argv", ["eon_driver.py", "--threshold", "5.0", "--default_element", "Fe123"]
    )
    with pytest.raises(SystemExit) as e:
        eon_driver.main()
    assert e.value.code == 100
    out, err = capsys.readouterr()
    assert "Invalid element symbol" in err


def test_main_empty_input_mock_pyacemaker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import sys

    monkeypatch.setattr(sys.stdin, "read", lambda size=0: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eon_driver.py",
            "--threshold",
            "5.0",
            "--potential",
            "None",
            "--default_element",
            "Fe",
            "--default_cell",
            "5.0",
        ],
    )

    # mock sys.modules to simulate pyacemaker import
    class DummyCalc:
        pass

    import sys

    class MockPyacemakerModule:
        pyacemaker = DummyCalc

    sys.modules["pyacemaker"] = MockPyacemakerModule()
    sys.modules["pyacemaker.calculator"] = MockPyacemakerModule()

    with pytest.raises(SystemExit) as e:
        eon_driver.main()
    assert e.value.code == 0


def test_main_with_potential(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import sys
    import typing

    monkeypatch.setattr(sys.stdin, "read", lambda size=0: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eon_driver.py",
            "--threshold",
            "5.0",
            "--potential",
            "test.yace",
            "--default_element",
            "Fe",
            "--default_cell",
            "5.0",
        ],
    )

    # mock sys.modules to simulate pyacemaker import
    class DummyCalc:
        def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
            pass

    class MockPyacemakerModule:
        pyacemaker = DummyCalc

    sys.modules["pyacemaker"] = MockPyacemakerModule()
    sys.modules["pyacemaker.calculator"] = MockPyacemakerModule()

    # Need to patch Atoms.get_potential_energy
    from ase import Atoms

    monkeypatch.setattr(Atoms, "get_potential_energy", lambda *args, **kwargs: -100.0)
    monkeypatch.setattr(Atoms, "get_forces", lambda *args, **kwargs: [[1.0, 2.0, 3.0]])

    # mock sys.exit to avoid pytest catching it
    monkeypatch.setattr(sys, "exit", lambda x=None: None)
    eon_driver.main()


def test_main_with_potential_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
):
    import sys
    import typing

    monkeypatch.setattr(sys.stdin, "read", lambda size=0: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eon_driver.py",
            "--threshold",
            "5.0",
            "--potential",
            "test.yace",
            "--default_element",
            "Fe",
            "--default_cell",
            "5.0",
        ],
    )

    # mock sys.modules to simulate pyacemaker import
    class DummyCalc:
        def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
            pass

    import sys

    class MockPyacemakerModule:
        pyacemaker = DummyCalc

    sys.modules["pyacemaker"] = MockPyacemakerModule()
    sys.modules["pyacemaker.calculator"] = MockPyacemakerModule()

    # Need to patch Atoms.get_potential_energy to raise exception
    from ase import Atoms

    monkeypatch.setattr(Atoms, "get_potential_energy", lambda *args, **kwargs: 1 / 0)

    from src.dynamics import eon_driver

    monkeypatch.setattr(eon_driver, "write_bad_structure", lambda *args, **kwargs: None)

    monkeypatch.setattr(sys, "exit", lambda x=None: None)

    eon_driver.main()


def test_read_coordinates_from_stdin_no_ase(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    import sys

    monkeypatch.setattr(sys.stdin, "read", lambda size=0: "")

    # Mock import error for ase
    import builtins
    import typing

    real_import = builtins.__import__

    def mock_import(
        name: str,
        globals_: typing.Any = None,
        locals_: typing.Any = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> typing.Any:
        if name in {"ase", "ase.io"}:
            msg = "ase is not available"
            raise ImportError(msg)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(SystemExit) as e:
        eon_driver.read_coordinates_from_stdin("Fe", 5.0)
    assert e.value.code == 100
    out, err = capsys.readouterr()
    assert "ase is not available" in err


def test_main_no_pyacemaker(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    import sys

    monkeypatch.setattr(sys.stdin, "read", lambda size=0: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eon_driver.py",
            "--threshold",
            "5.0",
            "--potential",
            "test.yace",
            "--default_element",
            "Fe",
            "--default_cell",
            "5.0",
        ],
    )

    # ensure pyacemaker cannot be imported
    if "pyacemaker" in sys.modules:
        del sys.modules["pyacemaker"]
    if "pyacemaker.calculator" in sys.modules:
        del sys.modules["pyacemaker.calculator"]

    import builtins
    import typing

    real_import = builtins.__import__

    def mock_import(
        name: str,
        globals_: typing.Any = None,
        locals_: typing.Any = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> typing.Any:
        if "pyacemaker" in name:
            msg = "pyacemaker is not available"
            raise ImportError(msg)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(SystemExit) as e:
        eon_driver.main()
    assert e.value.code == 100
    out, err = capsys.readouterr()
    assert "pyacemaker is not available" in err


def test_read_coordinates_from_stdin_with_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    content = '2\nLattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" Properties=species:S:1:pos:R:3\nFe 0.0 0.0 0.0\nFe 0.5 0.5 0.5\n2\nLattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" Properties=species:S:1:pos:R:3\nFe 0.1 0.1 0.1\nFe 0.6 0.6 0.6\n'
    import sys

    calls = [0]

    def mock_read(size=-1):
        if calls[0] == 0:
            calls[0] += 1
            return content
        return ""

    monkeypatch.setattr(sys.stdin, "read", mock_read)

    atoms = eon_driver.read_coordinates_from_stdin("Fe", 5.0)
    assert len(atoms) == 2
    assert atoms.get_positions()[0][0] == 0.1
