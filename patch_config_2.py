with open("src/domain_models/config.py") as f:
    content = f.read()

old_policy = """    fallback_metal_band_gap: float = Field(default=0.0, description="Fallback band gap for metals")
    fallback_metal_melting_point: float = Field(default=1500.0, description="Fallback melting point for metals")
    fallback_metal_bulk_modulus: float = Field(default=150.0, description="Fallback bulk modulus for metals")
    fallback_insulator_band_gap: float = Field(default=2.0, description="Fallback band gap for insulators")
    fallback_insulator_melting_point: float = Field(default=800.0, description="Fallback melting point for insulators")
    fallback_insulator_bulk_modulus: float = Field(default=50.0, description="Fallback bulk modulus for insulators")"""

new_policy = """    fallback_metal_band_gap: float = Field(default=0.0, ge=0.0, description="Fallback band gap for metals")
    fallback_metal_melting_point: float = Field(default=1500.0, gt=0.0, description="Fallback melting point for metals")
    fallback_metal_bulk_modulus: float = Field(default=150.0, gt=0.0, description="Fallback bulk modulus for metals")
    fallback_insulator_band_gap: float = Field(default=2.0, ge=0.0, description="Fallback band gap for insulators")
    fallback_insulator_melting_point: float = Field(default=800.0, gt=0.0, description="Fallback melting point for insulators")
    fallback_insulator_bulk_modulus: float = Field(default=50.0, gt=0.0, description="Fallback bulk modulus for insulators")"""

content = content.replace(old_policy, new_policy)

# Remove the field_validators we added in the previous cycle
import re

content = re.sub(r'    @field_validator\("fallback_metal_band_gap", "fallback_insulator_band_gap"\)[\s\S]*?return v\n', '', content)

with open("src/domain_models/config.py", "w") as f:
    f.write(content)
