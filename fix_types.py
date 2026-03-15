import re
with open("tutorials/UAT_AND_TUTORIAL.py", "r") as f:
    code = f.read()

# Replace the entire mock block cleanly to fix all issues at once
target = r"        # Proper Calculator Mock.*?sys\.modules\[\"pyacemaker\.calculator\"\] = .*?\n"
replacement = """        # Proper Calculator Mock
        class MockPaceCalculator:
            name = 'pace'
            implemented_properties: ClassVar[list[str]] = ['energy', 'forces', 'free_energy']
            def __init__(self, **kwargs: Any) -> None:
                self.results: dict[str, Any] = {}
            def calculate(self, atoms: Any = None, properties: Any = None, system_changes: Any = None) -> None:
                self.results = {}
                self.results['energy'] = -10.0 * len(atoms) if atoms else 0.0
                if atoms:
                    import numpy as np
                    self.results['forces'] = np.zeros((len(atoms), 3))

        # Setup module structure
        mock_module = type("pyacemaker", (), {"pyacemaker": True})
        mock_calc = type("calculator", (), {"PyACEMakerCalculator": MockPaceCalculator})
        setattr(mock_module, "calculator", mock_calc)
        sys.modules["pyacemaker"] = mock_module  # type: ignore
        sys.modules["pyacemaker.calculator"] = mock_calc  # type: ignore\n"""

code = re.sub(target, replacement, code, flags=re.DOTALL)

with open("tutorials/UAT_AND_TUTORIAL.py", "w") as f:
    f.write(code)
