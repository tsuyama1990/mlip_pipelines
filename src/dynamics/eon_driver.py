import argparse
import sys
import typing

if typing.TYPE_CHECKING:
    from ase import Atoms

import re
from pathlib import Path


def _raise_empty_stdin() -> None:
    msg = "Empty stdin"
    raise ValueError(msg)


def read_coordinates_from_stdin(default_element: str, default_cell: float) -> "Atoms":
    try:
        import io

        from ase import Atoms
        from ase.io import read
    except ImportError:
        sys.stderr.write("ase is not available.\n")
        sys.exit(100)

    lines: list[str] = sys.stdin.readlines()
    if not lines:
        sys.stderr.write("Empty stdin received, falling back to configurable default structure.\n")
        return Atoms(
            default_element,
            positions=[[0, 0, 0]],
            cell=[default_cell, default_cell, default_cell],
            pbc=True,
        )

    try:
        content = "".join(lines)
        atoms_obj = read(io.StringIO(content), format="extxyz")
    except Exception as e:
        sys.stderr.write(f"Failed to parse input stream: {e}\n")
        return Atoms(
            default_element,
            positions=[[0, 0, 0]],
            cell=[default_cell, default_cell, default_cell],
            pbc=True,
        )
    else:
        if isinstance(atoms_obj, list):
            return atoms_obj[-1]  # type: ignore
        return atoms_obj  # type: ignore


def write_bad_structure(path: str, atoms: "Atoms") -> None:
    # Security: path validation
    base_name = Path(path).name
    if not re.match(r"^[a-zA-Z0-9_.-]+$", base_name) or base_name != path:
        sys.stderr.write(f"Invalid path for writing bad structure: {path}\n")
        sys.exit(100)

    try:
        from ase.io import write

        write(path, atoms, format="extxyz")
    except Exception as e:
        sys.stderr.write(f"Failed to write bad structure: {e}\n")


def print_forces(forces: typing.Any) -> None:
    # Any represents an nx3 numpy array of forces from ASE
    for f in forces:
        sys.stdout.write(f"{f[0]} {f[1]} {f[2]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PACE Driver for EON")
    parser.add_argument("--threshold", type=float, required=True, help="Uncertainty threshold")
    parser.add_argument("--potential", type=str, default="None", help="Path to potential file")
    parser.add_argument(
        "--default_element", type=str, default="Fe", help="Default element for empty structures"
    )
    parser.add_argument("--default_cell", type=float, default=5.0, help="Default cell parameter")
    args = parser.parse_args()

    # Security Validation
    if not isinstance(args.threshold, float):
        sys.stderr.write("Invalid threshold type.\n")
        sys.exit(100)

    if args.potential != "None" and (
        not re.match(r"^[/a-zA-Z0-9_.-]+$", args.potential) or ".." in args.potential
    ):
        sys.stderr.write("Potential path contains invalid characters.\n")
        sys.exit(100)

    if not re.match(r"^[A-Za-z]+$", args.default_element):
        sys.stderr.write("Invalid element symbol.\n")
        sys.exit(100)

    try:
        from pyacemaker.calculator import pyacemaker
    except ImportError:
        sys.stderr.write("pyacemaker is not available.\n")
        sys.exit(100)

    atoms = read_coordinates_from_stdin(args.default_element, args.default_cell)

    if args.potential is None or args.potential == "None":
        sys.exit(0)

    calc = pyacemaker(args.potential)
    atoms.calc = calc

    try:
        energy = float(atoms.get_potential_energy())  # type: ignore
        forces = atoms.get_forces()  # type: ignore
        sys.stdout.write(f"{energy}\n")
        print_forces(forces)
    except Exception:
        write_bad_structure("bad_structure.cfg", atoms)
        sys.exit(100)


if __name__ == "__main__":
    main()
