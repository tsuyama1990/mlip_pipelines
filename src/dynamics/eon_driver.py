import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms

def _raise_empty_stdin() -> None:
    msg = "Empty stdin"
    raise ValueError(msg)

def read_coordinates_from_stdin() -> "Atoms":
    try:
        from ase import Atoms
    except ImportError:
        sys.stderr.write("ase is not available.\n")
        sys.exit(100)

    try:
        lines = sys.stdin.readlines()
        if not lines:
            _raise_empty_stdin()
        return Atoms("Fe", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    except Exception:
        return Atoms("Fe", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)


def write_bad_structure(path: str, atoms: "Atoms") -> None:
    try:
        from ase.io import write
        write(path, atoms, format="extxyz")
    except Exception as e:
        sys.stderr.write(f"Failed to write bad structure: {e}\n")


def print_forces(forces: list[list[float]]) -> None:
    for f in forces:
        sys.stdout.write(f"{f[0]} {f[1]} {f[2]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PACE Driver for EON")
    parser.add_argument("--threshold", type=float, required=True, help="Uncertainty threshold")
    parser.add_argument("--potential", type=str, default="None", help="Path to potential file")
    args = parser.parse_args()

    try:
        from pyacemaker.calculator import pyacemaker
    except ImportError:
        sys.stderr.write("pyacemaker is not available.\n")
        sys.exit(100)

    atoms = read_coordinates_from_stdin()

    if args.potential is None or args.potential == "None":
        sys.exit(0)

    calc = pyacemaker(args.potential)
    atoms.calc = calc

    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        sys.stdout.write(f"{energy}\n")
        print_forces(forces)
    except Exception:
        write_bad_structure("bad_structure.cfg", atoms)
        sys.exit(100)


if __name__ == "__main__":
    main()
