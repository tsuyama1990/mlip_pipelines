# CYCLE04 User Acceptance Testing

## 1. Test Scenarios

**Scenario ID: UAT-C04-01**
**Priority: High**
**Title: Validation of LAMMPS Soft-Start Resume Generation**
This scenario ensures that when the system attempts to resume an interrupted MD simulation, the `DynamicsEngine` correctly and safely injects a temporary Langevin thermostat into the LAMMPS input script. This is absolutely critical to prevent explosive atomic movements and immediate crashes caused by the suddenly shifted energy gradients of the newly updated MLIP potential.

**Scenario ID: UAT-C04-02**
**Priority: High**
**Title: Successful Parsing of Active Learning Halt Events vs Fatal Errors**
This scenario verifies that the Python orchestrator correctly distinguishes between a fatal physical crash (e.g., lost atoms due to extreme temperatures) and an intentional, safe uncertainty threshold halt triggered by our custom `fix watchdog`. This parsing accuracy prevents the system from crashing completely during a legitimate active learning cycle.

**Scenario ID: UAT-C04-03**
**Priority: Medium**
**Title: Generation of Two-Tier Watchdog Commands and Smooth Steps**
This scenario confirms that the LAMMPS input script accurately and perfectly reflects the configured `threshold_call_dft` and `smooth_steps` parameters defined by the user. This guarantees that short-lived, physically irrelevant thermal noise does not trigger unnecessary and massively expensive DFT calculations.

**Scenario ID: UAT-C04-04**
**Priority: Low**
**Title: Proper Handling of the Initial Data Reading Phase**
This scenario ensures that on the very first iteration of the active learning loop, the system correctly uses the standard `read_data` command to load the initial atomic geometry, proving the state machine correctly distinguishes between initialization and resumption.

## 2. Behavior Definitions

**GIVEN** the `DynamicsEngine` is specifically instructed to resume a simulation using a `.restart` file from a previous run
**WHEN** the `in.lammps` script string is programmatically generated
**THEN** the script must explicitly contain the exact command `read_restart`
**AND** it must contain a `fix langevin` command block followed immediately by a short `run 100` command to thermalize
**AND** it must contain an `unfix` command for that specific thermostat before the main, long-duration production run begins.

**GIVEN** a LAMMPS process exits abruptly with a non-zero error code indicating failure
**WHEN** the orchestrator carefully parses the resulting massive `log.lammps` file containing the specific string "AL_HALT"
**THEN** the engine must NOT raise a fatal `DynamicsHaltInterrupt` exception
**AND** it must return a Python dictionary indicating `halted: True` and cleanly containing the absolute path to the newly generated restart file for the next cycle.

**GIVEN** an `ActiveLearningThresholds` configuration loaded with `threshold_call_dft=0.08` and `smooth_steps=5`
**WHEN** the `DynamicsEngine` generates the initial `in.lammps` simulation script
**THEN** the script must contain a line exactly matching the string: `fix watchdog all halt 5 v_max_gamma > 0.08 error hard message "AL_HALT"`.

**GIVEN** the orchestrator initiates the very first cycle of the active learning loop (iteration 0)
**WHEN** the `DynamicsEngine` generates the initial script
**THEN** the script must strictly use the `read_data` command
**AND** it must completely lack any `read_restart` commands or soft-start `fix langevin` protocols.

**Scenario ID: UAT-C04-05**
**Priority: Low**
**Title: Generation of the Final Soft-Start Production Run Commands**
This scenario verifies that the input script cleanly transitions from the damped Langevin thermalization phase back to the purely NVE or NPT production physics ensemble, ensuring the long-term simulation correctly observes macroscopic diffusion and thermal properties without artificial dampening.

## 3. Extended Behavior Verification

**GIVEN** the `DynamicsEngine` generates a resume script containing a Langevin soft-start
**WHEN** the script is finalized and written to the temporary working directory
**THEN** the script must explicitly execute an `unfix soft_start` command
**AND** it must subsequently define the primary production thermostat or barostat (e.g., `fix npt`) before issuing the final, long `run` command string.

**GIVEN** the orchestrator receives a valid `AL_HALT` signal from the parsed log file
**WHEN** the system attempts to extract the specific atoms responsible for the halt
**THEN** the engine must correctly locate and parse the corresponding `dump.lammps` file
**AND** it must return the specific integer indices of the atoms whose `c_pace_gamma` value exceeded the defined `threshold_call_dft`.
