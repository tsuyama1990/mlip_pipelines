import re

with open("src/dynamics/eon_driver.py") as f:
    data = f.read()

# Fix _read_stdin_safely: remove regex
read_stdin_replacement = """def _read_stdin_safely(max_size: int) -> str:
    import sys
    content_chunks = []
    current_size = 0
    while True:
        chunk = sys.stdin.read(4096)
        if not chunk:
            break
        current_size += len(chunk)
        if current_size > max_size:
            sys.stderr.write(f"Input stream exceeds maximum allowed size ({max_size} bytes).\\n")
            sys.exit(100)

        content_chunks.append(chunk)
    return "".join(content_chunks)

def _validate_atoms(atoms_obj: "Atoms", chemical_symbols: list[str]) -> None:
    import sys
    import math
    if len(atoms_obj) > 10000:
        sys.stderr.write("Input exceeds maximum allowed number of atoms (10000).\\n")
        sys.exit(100)
    for atom in atoms_obj:
        if atom.symbol not in chemical_symbols:
            sys.stderr.write(f"Invalid chemical symbol found: {atom.symbol}\\n")
            sys.exit(100)
        for coord in atom.position:
            if math.isnan(coord) or math.isinf(coord) or not (-1e5 <= coord <= 1e5):
                sys.stderr.write("Invalid or out-of-bounds atomic coordinate.\\n")
                sys.exit(100)"""

data = re.sub(
    r"def _read_stdin_safely.*?sys\.exit\(100\)\n\n        content_chunks\.append\(chunk\)\n    return \"\"\.join\(content_chunks\)\n\n\ndef _validate_atoms.*?sys\.exit\(100\)",
    read_stdin_replacement,
    data,
    flags=re.DOTALL,
)

with open("src/dynamics/eon_driver.py", "w") as f:
    f.write(data)
