# CYCLE02: GUI to LAMMPS Compile Layer UAT Plan

## 1. Test Scenarios

### Scenario ID: UAT-02-01: Semantic Region Translation Validation
**Priority:** High
**Description:** Verify that the system correctly translates a set of visual atomic selections (provided via the API as a JSON array of 0-based integer indices representing, for instance, a "frozen" slab bottom) into a mathematically exact, 1-based LAMMPS `group` and `fix` script without ever requiring the user to define complex physical coordinate boxes. This ensures the critical "Visual and Semantic State Management" requirement is fully operational, replacing fragile geometric scripts with robust topological groupings.

### Scenario ID: UAT-02-02: Strict Rejection of Conflicting Semantic Tags
**Priority:** Critical
**Description:** Verify that the backend strictly rejects any GUI payload that attempts to assign contradictory physical properties to the exact same atom. For example, if a user's bulk selection in the 3D viewer accidentally overlaps due to a clumsy mouse movement, placing atom ID 5 simultaneously in both a `FREEZE` group and a `THERMOSTAT` group, the Pydantic validation layer must mathematically compute the intersection, catch this impossibility before LAMMPS execution, and return a highly descriptive HTTP 422 error to the user indicating exactly which atoms caused the conflict.

### Scenario ID: UAT-02-03: Security Against Out-Of-Bounds Atomic Indices
**Priority:** High
**Description:** Verify that the backend aggressively rejects a GUI payload containing an atom index that exceeds the total number of atoms currently loaded in the system's structural database. This is vital to prevent fatal `IndexError` exceptions during ASE numpy array manipulation and to preclude potential silent crashes or segmentation faults within the heavily optimized C++ LAMMPS engine due to referencing non-existent atomic memory addresses.

## 2. Behavior Definitions

### UAT-02-01: Semantic Region Translation Validation
**GIVEN** a running, healthy instance of the Adaptive-MLIP FastAPI backend and a predefined 100-atom structure successfully loaded into memory
**AND** a simulated GUI JSON payload containing a `SemanticRegionConfig` that firmly assigns `action: FREEZE` to `indices: [0, 1, 2]`
**WHEN** the payload is submitted to the `/config/submit` endpoint and the backend LAMMPS script generation utility is triggered
**THEN** the system seamlessly generates a text script containing the command `group frozen_atoms id 1 2 3` (strictly confirming the automated translation from 0-based Python indexing to 1-based LAMMPS indexing)
**AND** the generated script explicitly contains the exact command `fix freeze_fix frozen_atoms setforce 0.0 0.0 0.0`
**AND** the script contains absolutely no syntax formatting errors or illegal shell characters.

### UAT-02-02: Strict Rejection of Conflicting Semantic Tags
**GIVEN** a running instance of the Adaptive-MLIP FastAPI backend
**AND** a simulated GUI JSON payload containing a flawed `SemanticRegionConfig` that assigns `action: FREEZE` to `indices: [5, 6]` and simultaneously assigns `action: THERMOSTAT` to `indices: [4, 5]`
**WHEN** the conflicting payload is submitted via HTTP POST to the `/config/submit` endpoint
**THEN** the system's Pydantic validation layer immediately rejects the payload
**AND** responds to the client with an HTTP 422 Unprocessable Entity status code
**AND** the JSON error details explicitly indicate that atom index `5` possesses conflicting semantic assignments and must be resolved
**AND** the physics orchestrator is completely bypassed and never instantiated.

### UAT-02-03: Security Against Out-Of-Bounds Atomic Indices
**GIVEN** a predefined, small 10-atom structure loaded into the backend's active memory
**AND** a simulated GUI JSON payload erroneously assigning `action: ACTIVE` to the non-existent `indices: [9, 10, 11]`
**WHEN** the payload is submitted and the ASE backend tagging utility attempts to process the arrays
**THEN** a `ValueError` is aggressively raised by the bounds-checking logic *before* any tags are applied to the underlying numpy `Atoms` object
**AND** the resulting error payload clearly states that indices `10` and `11` are strictly out of bounds for the current 10-atom physical system.
