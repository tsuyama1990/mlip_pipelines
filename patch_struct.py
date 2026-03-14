with open("src/generators/structure_generator.py") as f:
    content = f.read()

content = content.replace(
    'bulk("Fe", crystalstructure="fcc", a=3.8, cubic=True)',
    'bulk("Fe", crystalstructure="fcc", a=self.config.fept_lattice_constant, cubic=True)',
)
content = content.replace(
    'bulk(\n                    "MgO", crystalstructure="rocksalt", a=4.21, basis=[[0, 0, 0], [0.5, 0.5, 0.5]]\n                )',
    'bulk(\n                    "MgO", crystalstructure="rocksalt", a=self.config.mgo_lattice_constant, basis=[[0, 0, 0], [0.5, 0.5, 0.5]]\n                )',
)
content = content.replace(
    "stack(mat1, mat2, axis=2, maxstrain=10.0)",
    "stack(mat1, mat2, axis=2, maxstrain=self.config.interface_max_strain)",
)

with open("src/generators/structure_generator.py", "w") as f:
    f.write(content)
