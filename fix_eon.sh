sed -i 's/def poll(self) -> int:/def poll(self) -> int | None:\n                return self.returncode/g' tests/unit/test_eon_wrapper.py
