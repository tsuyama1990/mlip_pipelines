with open("src/domain_models/config.py") as f:
    content = f.read()

content = content.replace(
    """    valid_interface_targets: list[str] = Field(
        default_factory=lambda: ["FePt", "MgO"],
        description="Whitelist of explicitly supported synthetic interface components",
    )""",
    """    valid_interface_targets: list[str] = Field(
        default_factory=lambda: ["FePt", "MgO"],
        description="Whitelist of explicitly supported synthetic interface components",
    )
    fept_lattice_constant: float = Field(default=3.8, description="Lattice constant for FePt")
    mgo_lattice_constant: float = Field(default=4.21, description="Lattice constant for MgO")
    interface_max_strain: float = Field(default=10.0, description="Max strain for stacking interfaces")""",
)

content = content.replace(
    """    strain_heavy_range: float = Field(
        default=0.15, description="strain_range for Strain-Heavy policy"
    )""",
    """    strain_heavy_range: float = Field(
        default=0.15, description="strain_range for Strain-Heavy policy"
    )
    uncertainty_variance_threshold: float = Field(
        default=1.0, description="Threshold for initial gamma variance to trigger cautious policy"
    )
    metal_band_gap_threshold: float = Field(
        default=0.1, description="Band gap threshold below which a material is considered a metal"
    )
    hard_material_bulk_modulus_threshold: float = Field(
        default=200.0, description="Bulk modulus threshold above which a material is considered hard"
    )
    fallback_metal_band_gap: float = Field(default=0.0, description="Fallback band gap for metals")
    fallback_metal_melting_point: float = Field(default=1500.0, description="Fallback melting point for metals")
    fallback_metal_bulk_modulus: float = Field(default=150.0, description="Fallback bulk modulus for metals")
    fallback_insulator_band_gap: float = Field(default=2.0, description="Fallback band gap for insulators")
    fallback_insulator_melting_point: float = Field(default=800.0, description="Fallback melting point for insulators")
    fallback_insulator_bulk_modulus: float = Field(default=50.0, description="Fallback bulk modulus for insulators")""",
)

with open("src/domain_models/config.py", "w") as f:
    f.write(content)
