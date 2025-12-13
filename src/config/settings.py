from typing import Literal, Optional, List, Dict
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

class MACESettings(BaseSettings):
    model_path: str = Field("medium", description="MACE model path or size (small, medium, large)")
    device: str = Field("cuda", description="Device to run on")
    default_dtype: Literal["float32", "float64"] = Field("float64", description="Default data type")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        if v == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        if v not in ["cpu", "cuda"]:
             # Allow specific cuda devices like cuda:0
             if v.startswith("cuda:"):
                 if not torch.cuda.is_available():
                     return "cpu"
                 return v
             raise ValueError("device must be 'cpu' or 'cuda'")
        return v

class RelaxationSettings(BaseSettings):
    fmax: float = Field(0.01, gt=0.0, description="Force convergence criterion (eV/A)")
    steps: int = Field(200, gt=0, description="Maximum number of optimization steps")
    optimizer: str = Field("LBFGS", description="Optimizer class name from ase.optimize")

class GeneratorSettings(BaseSettings):
    # This maps to the external AlloySystemConfig or similar
    # We will simplify and assume Alloy for this MVP as per prompt hints (target_element)
    type: str = Field("alloy", description="Generator type")
    target_element: str = Field("Si", description="Main element or formula")
    supercell_size: int = Field(2, description="Supercell expansion factor (applied to all dims)")
    vacancy_concentration: float = Field(0.0, description="Vacancy concentration")

    # Extra fields can be passed if needed
    elements: Optional[List[str]] = None

    @field_validator("elements", mode="before")
    @classmethod
    def set_elements_from_target(cls, v, info):
        # If elements not provided, derive from target_element
        if v is None and "target_element" in info.data:
            return [info.data["target_element"]]
        return v

class Settings(BaseSettings):
    mace: MACESettings = Field(default_factory=MACESettings)
    relax: RelaxationSettings = Field(default_factory=RelaxationSettings)
    generator: GeneratorSettings = Field(default_factory=GeneratorSettings)
    output_dir: Path = Field(Path("data/output"), description="Base output directory")
    random_seed: int = Field(42, description="Global random seed for reproducibility")

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")
