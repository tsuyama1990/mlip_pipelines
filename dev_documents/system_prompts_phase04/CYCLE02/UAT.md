# CYCLE02 User Acceptance Testing

## 1. Test Scenarios

**Scenario ID: UAT-C02-01**
**Priority: High**
**Title: Verification of Force Weight Assignments during Cluster Extraction**
This scenario ensures that when a highly uncertain cluster is extracted from a larger bulk structure, only the central "core" atoms receive a force weight of 1.0, while the protective "buffer" atoms correctly receive a weight of 0.0. This is absolutely critical to preventing inaccurate surface forces from polluting the MLIP training set and degrading the potential's accuracy.

**Scenario ID: UAT-C02-02**
**Priority: High**
**Title: Automatic Surface Passivation of Dangling Bonds**
This scenario verifies that the extraction module successfully identifies under-coordinated atoms on the surface of the extracted cluster and automatically attaches dummy passivating atoms (e.g., Hydrogen) to neutralize the local charge environment before passing the structure to the Oracle, ensuring SCF convergence.

**Scenario ID: UAT-C02-03**
**Priority: Medium**
**Title: Successful Pre-Relaxation with Frozen Core Constraints**
This scenario ensures that the MACE-driven pre-relaxation step successfully minimizes the energy of the buffer and passivating atoms, while the coordinates of the high-uncertainty core atoms remain absolutely frozen and unaltered, preserving the exact physical state that triggered the halt.

## 2. Behavior Definitions

**GIVEN** a bulk supercell structure and a specific target atom index representing an epicenter of high uncertainty
**WHEN** the `extract_intelligent_cluster` function is invoked with `core_radius=3.0` and `buffer_radius=5.0`
**THEN** the resulting `Atoms` object must contain a correctly sized numpy array named `force_weights`
**AND** all atoms within 3.0 Å of the target must have a precise weight of 1.0
**AND** all atoms strictly between 3.0 Å and 5.0 Å must have a precise weight of 0.0.

**GIVEN** an extracted atomic cluster containing a severely under-coordinated Oxygen atom at its physical boundary
**WHEN** the `_passivate_surface` function is executed with `passivation_element="H"`
**THEN** exactly one Hydrogen atom must be appended to the structure's coordinate array
**AND** its distance to the under-coordinated Oxygen atom must be physically reasonable (e.g., approximately 1.0 Å)
**AND** the `force_weight` of the newly added Hydrogen atom must be strictly set to 0.0.

**GIVEN** an extracted cluster with a clearly defined core region and buffer region
**WHEN** the `_pre_relax_buffer` function is executed using a surrogate MACE calculator
**THEN** the XYZ coordinates of all core atoms must exactly match their input coordinates down to machine precision
**AND** the XYZ coordinates of the buffer atoms must measurably differ from their input coordinates, indicating successful surrogate structural optimization occurred without disturbing the core.

**Scenario ID: UAT-C02-04**
**Priority: Low**
**Title: Graceful Handling of Edge Cases Near Cell Boundaries**
This scenario verifies that the spatial neighbor list queries used in `_extract_spherical_zones` correctly respect the minimum image convention when dealing with target atoms located extremely close to the periodic boundary of the supercell, ensuring no atoms are missed due to simple Euclidean distance fallacies.

**Scenario ID: UAT-C02-05**
**Priority: Low**
**Title: Validation of Fallback Mechanisms During Pre-Relaxation Failure**
This scenario tests the robust error handling within the pre-relaxation step. If the surrogate MACE model fails to converge the buffer atoms within the allocated 50 steps, the system must gracefully fall back to returning the unrelaxed (but still passivated) cluster, rather than crashing the entire orchestration loop and losing the halted trajectory state.

## 3. Extended Behavior Verification

**GIVEN** a bulk supercell where the target atom is located exactly at the coordinate origin (0, 0, 0)
**WHEN** the `extract_intelligent_cluster` function is invoked with a core radius of 4.0 Å
**THEN** the resulting `Atoms` object must correctly include atoms located near the opposite periodic boundaries (e.g., coordinates near $L_x, L_y, L_z$)
**AND** their calculated distances must accurately reflect the minimum image convention.

**GIVEN** an extracted cluster undergoing pre-relaxation via the surrogate MACE calculator
**WHEN** the LBFGS optimizer fails to converge the maximum forces below the 0.1 eV/Å threshold after the strict maximum of 50 steps
**THEN** the `extract_intelligent_cluster` function must NOT raise a fatal exception
**AND** it must return the partially relaxed `Atoms` object accompanied by a warning logged to the central orchestrator detailing the non-convergence event.
