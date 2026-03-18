import re

with open("src/core/orchestrator.py", "r") as f:
    text = f.read()

replacement_run = '''    def _run_exploration(
        self, current_pot: Path | None, tmp_work_dir: Path
    ) -> dict[str, Any] | str:
        try:
            strategy = self._decide_exploration_strategy()
            halt_info = self._execute_exploration(strategy, current_pot, tmp_work_dir)
            return self._detect_halt(halt_info)
        finally:
            pass'''

text = re.sub(
    r'    def _run_exploration\([\s\S]*?raise RuntimeError\(msg\) from e',
    replacement_run,
    text, count=1
)

with open("src/core/orchestrator.py", "w") as f:
    f.write(text)
