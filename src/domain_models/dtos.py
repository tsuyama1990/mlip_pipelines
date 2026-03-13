
from pydantic import BaseModel, ConfigDict, Field


class ExplorationStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r_md_mc: int = Field(
        default=0,
        ge=0,
        description="Ratio of MD steps per MC step. 0 means MD only.",
    )
    t_schedule: tuple[float, float, int] = Field(
        default=(300.0, 300.0, 10000),
        description="Temperature schedule: (T_start, T_end, steps).",
    )
    n_defects: int = Field(
        default=0,
        ge=0,
        description="Number of defects to introduce in the supercell.",
    )
    strain_range: float = Field(
        default=0.0,
        ge=0.0,
        description="Range of strain to apply for EOS/elasticity calculation (+/-).",
    )
    policy_type: str = Field(default="Standard-MD")
