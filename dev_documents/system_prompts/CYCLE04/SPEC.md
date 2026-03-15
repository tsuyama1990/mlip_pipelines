# CYCLE04 Specification

## 1. Summary

CYCLE04 tackles the critical "Time-Continuity Break" problem, which was a massive limitation of the Phase 01 architecture. Previously, when the long-running molecular dynamics (MD) simulation encountered a high uncertainty atomic configuration (triggering a halt event), the Python orchestrator would violently kill the LAMMPS process, retrain the machine learning model from scratch, and then completely restart the MD simulation from time $t=0$. This brute-force approach fundamentally prevented the observation of crucial long-timescale phenomena like atomic diffusion, slow structural phase transformations, or creeping crack propagation, rendering the pipeline useless for true High-Performance Computing (HPC) kinetic Monte Carlo (kMC) or prolonged nanosecond MD applications.

In this cycle, we will entirely rewrite the `DynamicsEngine` to implement a robust "Master-Slave Resume" paradigm. The engine will utilize the sophisticated FLARE-inspired two-tier uncertainty thresholds defined in CYCLE01. When the global uncertainty metric (`threshold_call_dft`) is breached for several consecutive timesteps (specifically ignoring ephemeral thermal noise spikes), the engine gracefully pauses the LAMMPS simulation. Crucially, it saves a perfect, bit-accurate binary `.restart` file. After the Python orchestrator performs the necessary cluster cutout, precise DFT calculation, and model updating (as handled in Cycles 2-5), the `DynamicsEngine` seamlessly re-initializes LAMMPS, loads the `.restart` file, dynamically re-applies the newly updated ACE MLIP, and continues the exact physical trajectory from time $t_{halt}$ without dropping a single microsecond of precious phase-space information.

## 2. System Architecture

The architecture shifts dramatically from treating LAMMPS as a disposable, single-use execution script to treating it as a stateful, highly resumable microservice. This is essential for achieving nanosecond simulation timescales.

The `DynamicsEngine` module will be extensively refactored to explicitly manage LAMMPS binary restart files and soft-start thermostats. It will generate highly specific `in.lammps` control scripts that utilize the `fix halt` command directly linked to the Pacemaker uncertainty output (`c_pace_gamma`). Upon a halt, the engine reads the LAMMPS log, confirms the specific physical reason for the halt, and parses the final dump file to extract the high-uncertainty atom indices for the subsequent intelligent cluster extraction phase.

### File Structure Modification

The engine code is deeply modified, relying on security utilities for safe parsing of the highly complex LAMMPS text outputs.

```text
.
├── src/
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── **dynamics_engine.py**  (Major Refactor: Implement master-slave resume and 2-tier thresholds)
│   │   ├── security_utils.py       (Utilize for parsing massive LAMMPS logs safely without regex bombs)
```

This strict architectural separation ensures the Orchestrator doesn't need to understand LAMMPS syntax. It simply calls `run_exploration`, and the engine returns a dictionary detailing whether the simulation converged, crashed fatally, or paused correctly for active learning.

## 3. Design Architecture

The design focuses heavily on managing the complex state machine of the molecular dynamics simulator via extremely robust input script generation and output log parsing.

### Domain Concepts and Constraints

1. **Stateful Continuation Logic:** The system must actively track the current active learning iteration. If `iteration == 0`, LAMMPS MUST start using the standard `read_data` command from an initial geometry file. If `iteration > 0`, LAMMPS MUST start using the `read_restart` command pointing exactly to the previous iteration's output file.
2. **Two-Tier Halt Logic (LAMMPS side):** The dynamically generated `in.lammps` script must define a compute for the ACE uncertainty (`c_pace_gamma`). It must define a specific variable tracking the maximum gamma across all atoms (`v_max_gamma`). Most importantly, it must implement a watchdog: `fix watchdog all halt ${smooth_steps} v_max_gamma > ${threshold_call_dft} error hard message "AL_HALT"`. The threshold values are injected directly from the strict `ActiveLearningThresholds` Pydantic model.
3. **Soft-Start Thermostat Protocol:** When resuming a simulation via `read_restart`, the newly updated potential will slightly alter the energy landscape. This can cause explosive forces if atoms are suddenly sitting on newly steepened gradients. The engine must automatically inject a temporary, highly damped `fix langevin` thermostat for the first ~100 steps of the resumed run to gently thermalize the system before returning to the original NVE/NPT dynamics.
4. **Halt Diagnostics and Differentiation:** Upon a LAMMPS crash or halt, the engine must expertly parse the massive `log.lammps` file. It must differentiate between a physical `Lost Atoms` error (which is fatal and means the simulation exploded) and our intentional `fix halt` trigger (which is an expected, positive active learning event).

## 4. Implementation Approach

The implementation centers on sophisticated string templating, safe subprocess management, and precise error handling.

1. **Input Script Generation:** Open `src/dynamics/dynamics_engine.py`. Radically modify the input script generation logic. Add a strict conditional block: if a `restart_file` path string is provided by the orchestrator, use the command `read_restart ${restart_file}`. Otherwise, use `read_data`.
2. **Watchdog Implementation:** Inject the following specific LAMMPS commands into the generated script, using secure Python string formatting to insert the configured `threshold_call_dft` and `smooth_steps` variables safely:
   ```lammps
   compute pace_gamma all pace ... gamma_mode=1
   variable max_gamma equal max(c_pace_gamma)
   fix watchdog all halt ${smooth_steps} v_max_gamma > ${threshold_call_dft} error hard message "AL_HALT"
   ```
3. **Soft-Start Logic:** If `read_restart` is utilized, inject a highly specific command block immediately after reading the restart file to engage the soft-start: `fix soft_start all langevin ${temp} ${temp} 0.1 48279`. Follow this immediately with a short `run 100` command, and crucially, an `unfix soft_start` command before beginning the main, long-duration production `run`.
4. **Subprocess Error Handling:** Wrap the main `subprocess.run(lammps)` call in a robust `try/except` block. If `subprocess.CalledProcessError` is raised by the OS, catch it gracefully. Read the final tail of the `log.lammps` file. If the exact string `"AL_HALT"` is present, return a structured dictionary indicating a successful active learning event occurred, along with the precise file paths to the generated `dump` and `restart` files. If the string is absent, immediately re-raise the error as a fatal `DynamicsHaltInterrupt`.

## 5. Test Strategy

Testing this cycle requires meticulously validating the intricate logic of LAMMPS input generation and log parsing without actually invoking the heavy LAMMPS binary during CI.

### Unit Testing Approach
We will thoroughly test the text-processing and script generation components within `tests/dynamics/test_dynamics_engine.py`.
- **Script Generation Test (Initial):** Initialize the engine cleanly without a restart file. Assert mathematically that the generated script contains the `read_data` string and completely lacks the `soft_start` langevin commands. Assert that the `fix watchdog` command accurately reflects the exact configured threshold values passed from the Pydantic models.
- **Script Generation Test (Resume):** Initialize the engine passing a dummy `previous.restart` file path. Assert that the script explicitly contains `read_restart previous.restart`. Assert that the `fix soft_start` and `unfix soft_start` commands are correctly present exactly before the main `run` command block.
- **Log Parsing Test:** Create a massive mock `log.lammps` string ending precisely with the `AL_HALT` message. Pass this string to the internal `_parse_halt_log` method. Assert that the method correctly identifies the halt as an intentional active learning trigger rather than a fatal crash, returning the correct boolean flag.

### Integration Testing Approach
We will securely simulate the subprocess execution environment.
- **Subprocess Mocking:** We will use `unittest.mock.patch('subprocess.run')` to fully intercept the LAMMPS OS call. We will configure the mock to deliberately raise a `subprocess.CalledProcessError` with a non-zero exit code. We will utilize Pytest's `tmp_path` fixture to write a dummy `log.lammps` file containing the `AL_HALT` string directly into the temporary workspace before the mock raises the error. We will assert that the `run_exploration` public method correctly catches the error, expertly parses the temporary log, and returns the expected structured response indicating a successful active learning interrupt, rather than panicking.
- **Side-Effect Management:** The mock subprocess prevents any actual LAMMPS binaries from executing, which is critical since LAMMPS is rarely installed on standard CI runners. Temporary directories strictly isolate all generated input scripts and dummy log files, ensuring the test suite cleans up after itself perfectly and leaves zero residual files on the developer's machine.
