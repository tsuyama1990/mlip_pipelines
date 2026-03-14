class OracleConvergenceError(Exception):
    """Raised when the DFT Oracle fails to converge."""


class DynamicsHaltInterrupt(Exception):  # noqa: N818
    """Raised when the Dynamics Engine halts due to high uncertainty."""
