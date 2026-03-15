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


def _read_stdin_safely(max_size: int) -> str:
    import sys

    content_chunks = []
    current_size = 0
    while True:
        chunk = sys.stdin.read(4096)
        if not chunk:
            break
        current_size += len(chunk)
        if current_size > max_size:
            sys.stderr.write(f"Input stream exceeds maximum allowed size ({max_size} bytes).\n")
            sys.exit(100)

        # Basic fast-fail validation for valid characters typically found in extxyz/xyz files
        # Only allow alphanumeric, spaces, and scientific notation characters (including quotes, often found in Lattice="")
        if not re.match(r'^[a-zA-Z0-9\s\.\-\+\:\=\_\"\']*$', chunk):
            sys.stderr.write("Invalid characters detected in input stream.\n")
            sys.exit(100)

        content_chunks.append(chunk)
    return "".join(content_chunks)


def _validate_atoms(atoms_obj: "Atoms", chemical_symbols: list[str]) -> None:
    import sys

    for atom in atoms_obj:
        if atom.symbol not in chemical_symbols:
            sys.stderr.write(f"Invalid chemical symbol found: {atom.symbol}\n")
            sys.exit(100)
        for coord in atom.position:
            if not (-1e5 <= coord <= 1e5):
                sys.stderr.write(f"Coordinate out of valid range: {coord}\n")
                sys.exit(100)


def read_coordinates_from_stdin(default_element: str, default_cell: float) -> "Atoms":
    import sys

    try:
        import io

        from ase.data import chemical_symbols
        from ase.io import read
    except ImportError:
        sys.stderr.write("ase is not available.\n")
        sys.exit(100)

    # Note: size limit should be configurable, defaulting to 10MB here.
    max_size = 10 * 1024 * 1024
    content = _read_stdin_safely(max_size)

    if not content.strip():
        sys.stderr.write("Empty stdin received.\n")
        sys.exit(100)

    try:
        atoms_obj = read(io.StringIO(content), format="extxyz")
    except Exception as e:
        sys.stderr.write(f"Failed to parse input stream: {e}\n")
        sys.exit(100)

    if isinstance(atoms_obj, list):
        if not atoms_obj:
            sys.stderr.write("No structures found in input.\n")
            sys.exit(100)
        atoms_obj = atoms_obj[-1]

    for atom in atoms_obj:
        if atom.symbol not in chemical_symbols:
            sys.stderr.write(f"Invalid chemical symbol found: {atom.symbol}\n")
            sys.exit(100)
        for coord in atom.position:
            import math

            if math.isnan(coord) or math.isinf(coord) or not (-1e5 <= coord <= 1e5):
                sys.stderr.write(f"Coordinate out of valid range or NaN/Inf: {coord}\n")
                sys.exit(100)

    return atoms_obj  # type: ignore


def write_bad_structure(path: str, atoms: "Atoms") -> None:
    import tempfile

    # Specific temporary directory whitelist
    allowed_dir = Path(tempfile.gettempdir()) / "mlip_bad_structures"
    allowed_dir.mkdir(parents=True, exist_ok=True)
    allowed_dir = allowed_dir.resolve(strict=True)

    try:
        resolved_path = Path(path).resolve(strict=False)
        if not resolved_path.is_relative_to(allowed_dir):
            sys.stderr.write(f"Path outside allowed directory: {path}\n")
            sys.exit(100)
    except Exception as e:
        sys.stderr.write(f"Path resolution error: {e}\n")
        sys.exit(100)

    base_name = resolved_path.name
    if not re.match(r"^[a-zA-Z0-9_.-]+$", base_name):
        sys.stderr.write(f"Invalid filename: {base_name}\n")
        sys.exit(100)

    try:
        from ase.io import write

        write(str(resolved_path), atoms, format="extxyz")
    except Exception as e:
        sys.stderr.write(f"Failed to write bad structure: {e}\n")


def print_forces(forces: typing.Any) -> None:
    # Any represents an nx3 numpy array of forces from ASE
    for f in forces:
        sys.stdout.write(f"{f[0]} {f[1]} {f[2]}\n")


def _validate_args(args: argparse.Namespace) -> tuple[float, float]:
    try:
        threshold = float(args.threshold)
    except ValueError:
        sys.stderr.write("Invalid threshold type or value.\n")
        sys.exit(100)

    if threshold < 0:
        sys.stderr.write("Invalid threshold type or value.\n")
        sys.exit(100)

    if args.potential != "None" and (
        not re.match(r"^[/a-zA-Z0-9_.-]+$", args.potential) or ".." in args.potential
    ):
        sys.stderr.write("Potential path contains invalid characters.\n")
        sys.exit(100)

    try:
        from ase.data import chemical_symbols
    except ImportError:
        sys.stderr.write("ase is not available.\n")
        sys.exit(100)

    if args.default_element not in chemical_symbols:
        sys.stderr.write("Invalid element symbol.\n")
        sys.exit(100)

    try:
        default_cell = float(args.default_cell)
    except ValueError:
        sys.stderr.write("Invalid default_cell format.\n")
        sys.exit(100)

    if default_cell <= 0:
        sys.stderr.write("Invalid default_cell format.\n")
        sys.exit(100)

    return threshold, default_cell


def main() -> None:
    parser = argparse.ArgumentParser(description="PACE Driver for EON")
    parser.add_argument("--threshold", type=str, required=True, help="Uncertainty threshold")
    parser.add_argument("--potential", type=str, default="None", help="Path to potential file")
    parser.add_argument(
        "--default_element", type=str, default="Fe", help="Default element for empty structures"
    )
    parser.add_argument("--default_cell", type=str, default="5.0", help="Default cell parameter")
    args = parser.parse_args()

    threshold, default_cell = _validate_args(args)

    try:
        from pyacemaker.calculator import pyacemaker
    except ImportError:
        sys.stderr.write("pyacemaker is not available.\n")
        sys.exit(100)

    atoms = read_coordinates_from_stdin(args.default_element, default_cell)

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
