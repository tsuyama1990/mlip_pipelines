with open("src/domain_models/config.py") as f:
    content = f.read()

# Replace StructureGeneratorConfig
old_struct = """class StructureGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stdev: float = Field(default=0.05, ge=0.0, description="Standard deviation for rattling")
    seed_base: int = Field(default=42, description="Base seed for reproducibility")
    valid_interface_targets: list[str] = Field(
        default_factory=lambda: ["FePt", "MgO"],
        description="Whitelist of explicitly supported synthetic interface components",
    )
    fept_lattice_constant: float = Field(default=3.8, description="Lattice constant for FePt")
    mgo_lattice_constant: float = Field(default=4.21, description="Lattice constant for MgO")
    interface_max_strain: float = Field(default=10.0, description="Max strain for stacking interfaces")"""

new_struct = """class StructureGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stdev: float = Field(default=0.05, ge=0.0, description="Standard deviation for rattling")
    seed_base: int = Field(default=42, description="Base seed for reproducibility")
    valid_interface_targets: list[str] = Field(
        default=["FePt", "MgO"],
        description="Whitelist of explicitly supported synthetic interface components",
    )
    fept_lattice_constant: float = Field(default=3.8, description="Lattice constant for FePt")
    mgo_lattice_constant: float = Field(default=4.21, description="Lattice constant for MgO")
    interface_max_strain: float = Field(default=10.0, description="Max strain for stacking interfaces")"""

content = content.replace(old_struct, new_struct)

with open("src/domain_models/config.py", "w") as f:
    f.write(content)
