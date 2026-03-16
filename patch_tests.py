with open("tests/unit/test_eon_driver.py") as f:
    data = f.read()

# Fix the test
replace_from = """    try:
        eon_driver.write_bad_structure("../bad.cfg", Atoms("Fe"))
    except SystemExit as e:
        assert e.code == 100
        out, err = capsys.readouterr()
        assert "Invalid filename" in err"""

replace_to = """    with pytest.raises(SystemExit) as e:
        eon_driver.write_bad_structure("../bad.cfg", Atoms("Fe"))
    assert e.value.code == 100
    out, err = capsys.readouterr()
    assert "Invalid filename" in err"""

data = data.replace(replace_from, replace_to)

with open("tests/unit/test_eon_driver.py", "w") as f:
    f.write(data)
