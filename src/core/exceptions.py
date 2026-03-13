class MLIPPipelineError(Exception):
    """Base class for all mlip-pipelines exceptions."""


class InvalidStructureError(MLIPPipelineError):
    """Raised when a structure geometry is physically invalid."""


class OracleComputationError(MLIPPipelineError):
    """Raised when the DFT Oracle fails to compute properties (e.g. SCF non-convergence)."""


class UncertaintyInterruptError(MLIPPipelineError):
    """Raised when dynamics simulation hits an unphysical extrapolation region (gamma > threshold)."""


class DynamicsHaltInterruptError(UncertaintyInterruptError):
    """Raised specifically when a watchdog gracefully halts a dynamics run."""


class OracleConvergenceError(OracleComputationError):
    """Raised when the Oracle specifically fails to converge after maximum retries."""
