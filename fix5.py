import re

with open("tests/uat/verify_cycle_06_domain_logic.py", "r") as f:
    text = f.read()

# Fix Orchestrator state resumption logic in test_scenario_1
replacement_scen = '''    def test_scenario_1(DummyConfig, Orchestrator, tmp_path):
        import tempfile
        # Scenario ID: UAT-C06-01: HPC Wall-Time Job Kill Recovery and State Resumption
        print("Testing UAT-C06-01: HPC Kill Recovery")
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td).resolve()
            config = DummyConfig(str(temp_dir))

            orch1 = Orchestrator(config)
            # Simulate a run that completed Phase1 and is killed
            orch1.checkpoint.set_state("CURRENT_PHASE", "PHASE2_VALIDATION")
            orch1.checkpoint.set_state("CURRENT_ITERATION", 5)

            # "Restart" the process
            orch2 = Orchestrator(config)

            assert orch2.checkpoint.get_state("CURRENT_PHASE") == "PHASE2_VALIDATION", (
                "State did not persist!"
            )
            assert orch2.iteration == 5, (
                f"Iteration counter did not resume correctly! It is {orch2.iteration}"
            )
            print("✓ Orchestrator successfully resumed from database checkpoint")
            return orch1, orch2'''

text = re.sub(
    r'    def test_scenario_1\(DummyConfig, Orchestrator, tmp_path\):[\s\S]*?return orch1, orch2',
    replacement_scen,
    text, count=1
)

with open("tests/uat/verify_cycle_06_domain_logic.py", "w") as f:
    f.write(text)
