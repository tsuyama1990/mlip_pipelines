# CYCLE02 UAT: Structure Generator & Policy Engine

## Test Scenarios

### Scenario 1: Adaptive Exploration Policy Evaluation
**ID**: `UAT-02-01`
**Priority**: High
**Description**: Verify that the `AdaptiveExplorationPolicyEngine` correctly evaluates different material profiles and outputs the appropriate `ExplorationStrategy`. When presented with a metallic system ($E_g \approx 0$), it should prioritize a High-MC strategy to sample diffusion. When presented with an insulating system ($E_g > 0$), it should prioritize a Defect-Driven strategy to sample structural distortions.

### Scenario 2: Robust Interface Generation
**ID**: `UAT-02-02`
**Priority**: Medium
**Description**: Verify that the `StructureGenerator` successfully creates a complex synthetic interface (e.g., FePt on MgO) based on the `InterfaceTarget` configuration. It must securely validate the requested elements against a whitelist and gracefully handle lattice mismatches using ASE's stacking tools.

### Scenario 3: OOM Protection During Candidate Generation
**ID**: `UAT-02-03`
**Priority**: High
**Description**: Verify that the `StructureGenerator.generate_local_candidates` method strictly limits the size of the input structure (`s0`) and the number of candidates generated (`n`) to prevent Out-Of-Memory (OOM) errors and Denial of Service (DoS) attacks from excessively large cluster requests.

## Behavior Definitions

### UAT-02-01: Adaptive Exploration Policy Evaluation
```gherkin
GIVEN a `MaterialFeatures` object representing a multi-component metal ($E_g \approx 0$)
WHEN the policy engine evaluates these features via `decide_policy()`
THEN the returned `ExplorationStrategy` should have an `md_mc_ratio` greater than 0
AND the `policy_name` should indicate "High-MC Policy"

GIVEN a `MaterialFeatures` object representing an insulator ($E_g > 0$)
WHEN the policy engine evaluates these features via `decide_policy()`
THEN the returned `ExplorationStrategy` should have `n_defects` greater than 0
AND the `policy_name` should indicate "Defect-Driven Policy"
```

### UAT-02-02: Robust Interface Generation
```gherkin
GIVEN the `SystemConfig` contains a valid `InterfaceTarget` for elements `FePt` and `MgO`
AND these elements are present in the `valid_interface_targets` whitelist
WHEN the Orchestrator triggers `generate_interface()` during iteration 0
THEN the `StructureGenerator` should successfully create a combined `Atoms` object
AND the object should contain both `Fe`, `Pt`, `Mg`, and `O` atoms
AND the resulting structure should be saved to `initial_structure.extxyz` in the run directory.

GIVEN an `InterfaceTarget` requests an unsupported element (e.g., `Unobtainium`)
WHEN `generate_interface()` is called
THEN a `ValueError` should be raised immediately
AND the generation process should halt.
```

### UAT-02-03: OOM Protection During Candidate Generation
```gherkin
GIVEN a massively large `Atoms` object (e.g., 50,000 atoms) representing a highly uncertain structure
WHEN `generate_local_candidates()` is called with this object
THEN the method should immediately raise a `ValueError` indicating the structure is too large for rattling
AND no candidates should be generated in memory.

GIVEN a moderately large `Atoms` object (e.g., 2,000 atoms) and a request for `n=200` candidates
WHEN `generate_local_candidates()` is called
THEN the method should automatically scale down the number of generated candidates to a safe maximum (e.g., `max(1, n // 10)`)
AND the returned list of rattled structures should not exceed this safe limit.
```