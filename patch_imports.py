def fix_imports(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    imports = []
    other_lines = []

    for line in lines:
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
        else:
            other_lines.append(line)

    with open(filepath, "w") as f:
        f.writelines(imports)
        f.writelines(other_lines)


fix_imports("tests/e2e/test_skeleton.py")
fix_imports("tests/uat/test_tutorial.py")
fix_imports("tests/unit/test_adaptive_policy.py")
fix_imports("tests/unit/test_structure_generator.py")
