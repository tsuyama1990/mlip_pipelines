with open("src/generators/adaptive_policy.py") as f:
    content = f.read()

old_try = """    def _try_load_mlip(self) -> None:
        import importlib.util
        if importlib.util.find_spec("matgl") is not None and importlib.util.find_spec("chgnet") is not None:
            msg = "Full MLIP inference not implemented in UAT mode"
            raise NotImplementedError(msg)
        msg2 = "matgl or chgnet not available"
        raise ImportError(msg2)"""

new_try = """    def _try_load_mlip(self) -> None:
        import importlib.util
        has_matgl = importlib.util.find_spec("matgl") is not None
        has_chgnet = importlib.util.find_spec("chgnet") is not None

        if not has_matgl and not has_chgnet:
            msg = "Neither matgl nor chgnet are available."
            raise ImportError(msg)

        msg2 = "MLIPs found, but full inference not implemented in UAT mode."
        raise NotImplementedError(msg2)"""

content = content.replace(old_try, new_try)

with open("src/generators/adaptive_policy.py", "w") as f:
    f.write(content)
