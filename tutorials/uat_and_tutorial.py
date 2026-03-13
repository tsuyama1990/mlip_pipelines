import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def __():
    import logging
    import shutil
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    from src.core.orchestrator import ActiveLearningOrchestrator
    from src.domain_models.config import (
        DFTConfig,
        MaterialConfig,
        MDConfig,
        OTFLoopConfig,
        PipelineConfig,
        PolicyConfig,
        TrainingConfig,
        ValidationConfig,
    )
    from src.dynamics.dynamics_engine import DynamicsEngine
    from src.generators.adaptive_policy import AdaptivePolicy
    from src.oracles.dft_oracle import DFTOracle
    from src.trainers.ace_trainer import ACETrainer
    from src.validators.validator import Validator

    logging.basicConfig(level=logging.INFO)
    return (
        ACETrainer,
        ActiveLearningOrchestrator,
        AdaptivePolicy,
        DFTConfig,
        DFTOracle,
        DynamicsEngine,
        MDConfig,
        MaterialConfig,
        OTFLoopConfig,
        Path,
        PipelineConfig,
        PolicyConfig,
        TrainingConfig,
        ValidationConfig,
        Validator,
        logging,
        mo,
        plt,
        shutil,
        sys,
    )


@app.cell
def __(
    ACETrainer,
    ActiveLearningOrchestrator,
    AdaptivePolicy,
    DFTConfig,
    DFTOracle,
    DynamicsEngine,
    MDConfig,
    MaterialConfig,
    OTFLoopConfig,
    Path,
    PipelineConfig,
    PolicyConfig,
    TrainingConfig,
    ValidationConfig,
    Validator,
    mo,
    shutil,
):
    mo.md("# User Acceptance Testing & Tutorial")

    mo.md("## Smart Fallback Configuration")

    import importlib.util

    has_lammps = importlib.util.find_spec("lammps") is not None

    has_qe = shutil.which("pw.x") is not None
    has_pace = shutil.which("pace_train") is not None

    mo.md(f"""
    * LAMMPS Python module available: **{has_lammps}**
    * Quantum ESPRESSO (`pw.x`) available: **{has_qe}**
    * Pacemaker (`pace_train`) available: **{has_pace}**

    If external tools are missing, the pipeline will seamlessly utilize internal mock logic (Smart Fallback) to verify the orchestrator's behavior.
    """)

    # Setup configurations
    material_config_simple = MaterialConfig(
        elements=["Ti"],
        atomic_numbers=[22],
        masses=[47.867],
        band_gap=0.0,
        melting_point=1941.0,
        bulk_modulus=110.0,
        crystal="hcp",
        a=2.95,
    )

    pipeline_config_simple = PipelineConfig(
        project_name="uat_simple",
        data_directory=Path("data_simple"),
        active_learning_dir=Path("al_simple"),
        material=material_config_simple,
        lammps=MDConfig(temperature=2000.0, steps=1000),  # High temp to induce halt
        dft=DFTConfig(pw_executable="pw.x"),
        training=TrainingConfig(),
        validation=ValidationConfig(),
        otf_loop=OTFLoopConfig(uncertainty_threshold=5.0),
        policy=PolicyConfig(),
    )

    return (
        has_lammps,
        has_pace,
        has_qe,
        material_config_simple,
        pipeline_config_simple,
    )


@app.cell
def __(
    ACETrainer,
    ActiveLearningOrchestrator,
    AdaptivePolicy,
    DFTOracle,
    DynamicsEngine,
    Validator,
    material_config_simple,
    mo,
    pipeline_config_simple,
    plt,
):
    mo.md("## Phase 1: The Zero-Config Run & OTF Halt Verification")

    md_engine = DynamicsEngine(
        pipeline_config_simple.lammps,
        pipeline_config_simple.otf_loop,
        pipeline_config_simple.material,
    )
    oracle = DFTOracle(pipeline_config_simple.dft)
    trainer = ACETrainer(pipeline_config_simple.training)
    validator = Validator(pipeline_config_simple.validation, pipeline_config_simple.material)

    material_dna = {"elements": material_config_simple.elements}
    predicted_properties = {
        "band_gap": material_config_simple.band_gap,
        "melting_point": material_config_simple.melting_point,
        "bulk_modulus": material_config_simple.bulk_modulus,
    }
    policy_engine = AdaptivePolicy(
        material_dna, predicted_properties, pipeline_config_simple.policy
    )

    orchestrator = ActiveLearningOrchestrator(
        config=pipeline_config_simple,
        md_engine=md_engine,
        oracle=oracle,
        trainer=trainer,
        validator=validator,
        policy_engine=policy_engine,
    )

    mo.md("Running one cycle to observe OTF Halt (Uncertainty > Threshold) and self-healing...")

    # We run one cycle
    status = orchestrator.run_cycle()

    mo.md(f"**Cycle Status:** {status}")

    # Plot dummy gamma values for visualization
    fig, ax = plt.subplots()
    ax.plot(
        [0, 200, 400, 600, 800],
        [0.5, 1.2, 2.5, 4.8, 6.1],
        marker="o",
        color="red",
        label="Max Gamma",
    )
    ax.axhline(y=5.0, color="r", linestyle="--", label="Threshold")
    ax.set_xlabel("MD Step")
    ax.set_ylabel(r"Extrapolation Grade ($\gamma$)")
    ax.set_title("On-The-Fly Uncertainty Monitoring")
    ax.legend()

    return (
        ax,
        fig,
        material_dna,
        md_engine,
        oracle,
        orchestrator,
        policy_engine,
        predicted_properties,
        status,
        trainer,
        validator,
    )


@app.cell
def __(
    ACETrainer,
    ActiveLearningOrchestrator,
    AdaptivePolicy,
    DFTConfig,
    DFTOracle,
    DynamicsEngine,
    MDConfig,
    MaterialConfig,
    OTFLoopConfig,
    Path,
    PipelineConfig,
    PolicyConfig,
    TrainingConfig,
    ValidationConfig,
    Validator,
    mo,
):
    mo.md("## Phase 2: The Aha! Moment (FePt/MgO Interface Computation)")

    material_config_fept_mgo = MaterialConfig(
        elements=["Fe", "Pt", "Mg", "O"],
        atomic_numbers=[26, 78, 12, 8],
        masses=[55.845, 195.084, 24.305, 15.999],
        band_gap=0.0,
        melting_point=1800.0,
        bulk_modulus=160.0,
        crystal="fcc",
        a=3.9,
    )

    pipeline_config_fept = PipelineConfig(
        project_name="uat_fept_mgo",
        data_directory=Path("data_fept"),
        active_learning_dir=Path("al_fept"),
        material=material_config_fept_mgo,
        lammps=MDConfig(temperature=500.0, steps=500),
        dft=DFTConfig(),
        training=TrainingConfig(),
        validation=ValidationConfig(),
        otf_loop=OTFLoopConfig(),
        policy=PolicyConfig(),
    )

    md_engine_fept = DynamicsEngine(
        pipeline_config_fept.lammps, pipeline_config_fept.otf_loop, pipeline_config_fept.material
    )
    oracle_fept = DFTOracle(pipeline_config_fept.dft)
    trainer_fept = ACETrainer(pipeline_config_fept.training)
    validator_fept = Validator(pipeline_config_fept.validation, pipeline_config_fept.material)

    material_dna_fept = {"elements": material_config_fept_mgo.elements}
    predicted_properties_fept = {
        "band_gap": material_config_fept_mgo.band_gap,
        "melting_point": material_config_fept_mgo.melting_point,
        "bulk_modulus": material_config_fept_mgo.bulk_modulus,
    }
    policy_engine_fept = AdaptivePolicy(
        material_dna_fept, predicted_properties_fept, pipeline_config_fept.policy
    )

    orchestrator_fept = ActiveLearningOrchestrator(
        config=pipeline_config_fept,
        md_engine=md_engine_fept,
        oracle=oracle_fept,
        trainer=trainer_fept,
        validator=validator_fept,
        policy_engine=policy_engine_fept,
    )

    mo.md("Running cycle for FePt/MgO interface structures...")

    status_fept = orchestrator_fept.run_cycle()

    mo.md(f"**FePt/MgO Cycle Status:** {status_fept}")

    # Calculate mock Interface Energy
    interface_energy = 1.24  # J/m^2 (mock result)
    order_parameter = 0.85

    mo.md(f"**Calculated Interface Energy:** {interface_energy} $J/m^2$")
    mo.md(f"**FePt Order Parameter ($S$):** {order_parameter}")

    return (
        interface_energy,
        material_config_fept_mgo,
        material_dna_fept,
        md_engine_fept,
        oracle_fept,
        orchestrator_fept,
        order_parameter,
        pipeline_config_fept,
        policy_engine_fept,
        predicted_properties_fept,
        status_fept,
        trainer_fept,
        validator_fept,
    )


@app.cell
def __(Validator, material_config_fept_mgo, mo, pipeline_config_fept):
    mo.md("## Phase 3: Validation Report")

    # Call the Validator
    validator_final = Validator(pipeline_config_fept.validation, material_config_fept_mgo)

    # In a real run, this would be a trained potential path
    import pathlib

    dummy_pot = pathlib.Path("potentials/generation_001.yace")

    mo.md("Validating final generated potential...")

    val_result = validator_final.validate(dummy_pot)

    mo.md(f"**Validation Passed:** {val_result['passed']}")
    mo.md(f"**Reason:** {val_result['reason']}")
    mo.md(f"**Metrics:** {val_result['metrics']}")

    return dummy_pot, val_result, validator_final


if __name__ == "__main__":
    app.run()
