import re

with open("tutorials/UAT_AND_TUTORIAL.py", "r") as f:
    code = f.read()

# Replace class MockPaceCalculator block
target = r"    class MockPaceCalculator\(Calculator\):.*?\n.*?self.results\['forces'\] = np.zeros\(\(len\(atoms\), 3\)\)\n"
replacement = """    class MockPaceCalculator:
        implemented_properties: ClassVar[list[str]] = ['energy', 'forces', 'free_energy']
        name = "pace"

        def __init__(self, **kwargs: Any) -> None:
            self.results: dict[str, Any] = {}

        def calculate(self, atoms: Any = None, properties: Any = None, system_changes: Any = None) -> None:
            self.results = {}
            if atoms is not None:
                import numpy as np
                self.results['energy'] = -10.0 * len(atoms)
                self.results['forces'] = np.zeros((len(atoms), 3))
"""

code = re.sub(target, replacement, code, flags=re.DOTALL)

with open("tutorials/UAT_AND_TUTORIAL.py", "w") as f:
    f.write(code)
