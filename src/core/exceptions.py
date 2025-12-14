class MLIPPipelineError(Exception):
    """Base exception for MLIP pipeline"""
    pass

class StructureGenerationError(MLIPPipelineError):
    """Failed to generate valid structure"""
    pass

class InvalidStructureError(MLIPPipelineError):
    """Structure violates physical constraints"""
    pass

class ConvergenceError(MLIPPipelineError):
    """Optimization failed to converge"""
    pass

class OracleComputationError(MLIPPipelineError):
    """Oracle calculation failed"""
    pass

class UncertaintyInterrupt(Exception):
    def __init__(self, atoms, uncertainty):
        self.atoms = atoms
        self.uncertainty = uncertainty
        super().__init__(f"Uncertainty exceeded threshold: {uncertainty}")
