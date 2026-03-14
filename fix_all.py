def fix_file(filepath):
    with open(filepath) as f:
        content = f.read()

    # Remove all duplicated appends by finding the first import block and keeping only the first definitions

    # Actually, it's easier to just git checkout origin/main if I can
    import subprocess

    subprocess.run(["git", "checkout", "origin/main", "--", filepath])


fix_file("tests/e2e/test_skeleton.py")
fix_file("tests/uat/test_tutorial.py")
fix_file("tests/unit/test_adaptive_policy.py")
fix_file("tests/unit/test_structure_generator.py")
