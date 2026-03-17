# CYCLE03 User Acceptance Testing

## 1. Test Scenarios

**Scenario ID: UAT-C03-01**
**Priority: High**
**Title: Zero-Shot Distillation Structure Acceptance and Fast-Tracking**
This scenario verifies that atomic structures evaluated by the MACE foundation model that fall confidently below the defined uncertainty threshold are immediately and safely accepted into the Phase 1 training set. This proves the system correctly fast-tracks simple configurations without invoking the immensely expensive DFT fallback oracle, proving the zero-shot distillation performance gains.

**Scenario ID: UAT-C03-02**
**Priority: High**
**Title: Tiered Oracle Routing of High-Uncertainty Structures to DFT**
This scenario ensures that the `TieredOracle` successfully and reliably identifies structures with a MACE uncertainty score that exceeds the safety threshold. It confirms the system correctly queues these difficult, out-of-distribution structures for rigorous calculation by the `DFTManager`, verifying the active learning loop's primary safety mechanism against learning incorrect physics.

**Scenario ID: UAT-C03-03**
**Priority: Medium**
**Title: Validation of MACE Epistemic Uncertainty Extraction and Normalization**
This scenario confirms that the `MACEManager` successfully extracts the raw epistemic uncertainty metric (e.g., committee variance or latent distance) from the underlying complex MACE PyTorch calculator. It verifies the manager normalizes this metric if necessary and attaches it securely and immutably to the `Atoms.info` dictionary for downstream routing decisions.

**Scenario ID: UAT-C03-04**
**Priority: Low**
**Title: Safety Fallback for Missing Uncertainty Metrics**
This scenario ensures that if the foundation model silently fails to produce an uncertainty metric due to an internal error or unexpected atomic configuration, the `TieredOracle` fails safely. It must route the structure to the DFT oracle rather than blindly trusting the MACE predicted forces without a confidence score.

## 2. Behavior Definitions

**GIVEN** the `TieredOracle` is correctly initialized with a `threshold_call_dft` of 0.05
**WHEN** a batch of 10 structurally diverse atomic configurations is processed, and the MACE primary oracle confidently assigns an uncertainty of 0.01 to all 10 structures
**THEN** the `TieredOracle` must return the 10 calculated structures immediately to the orchestrator
**AND** the `DFTManager`'s `compute_batch` method must be called exactly 0 times during this operation.

**GIVEN** the `TieredOracle` is correctly initialized with a `threshold_call_dft` of 0.05
**WHEN** a massive batch of 100 structures is processed, and the MACE oracle assigns an uncertainty of 0.10 to exactly 12 specific highly-distorted structures
**THEN** the `TieredOracle` must ultimately return exactly 100 calculated structures
**AND** the `DFTManager`'s `compute_batch` method must be called exactly 1 time with a payload of exactly 12 structures containing those specific distorted configurations.

**GIVEN** a valid `ase.Atoms` object passed to the `MACEManager` for processing
**WHEN** the `compute_batch` method is fully executed
**THEN** the resulting `Atoms` object must firmly contain a float value accessible via `atoms.info['mace_uncertainty']`
**AND** the `atoms.info['energy']` scalar field and `atoms.arrays['forces']` multi-dimensional arrays must be fully populated with physical float values.

**GIVEN** the `TieredOracle` processes a structure where the primary oracle fails to populate the `mace_uncertainty` key
**WHEN** the routing logic evaluates the structure
**THEN** the system must strictly interpret the missing key as infinite uncertainty
**AND** the structure must be forcefully routed to the `fallback_queue` for DFT evaluation.

**Scenario ID: UAT-C03-05**
**Priority: Low**
**Title: Validation of Fallback Queue Processing**
This scenario ensures that the `fallback_queue` populated by the `TieredOracle` correctly maintains the atomic properties and metadata of the high-uncertainty structures before routing them to the DFT execution engine, preventing any data loss during the routing handoff.

## 3. Extended Behavior Verification

**GIVEN** a highly uncertain `ase.Atoms` object that lacks confidence in its MACE prediction
**WHEN** the `TieredOracle` processes this object
**THEN** the structure must be placed precisely into the `fallback_queue` array
**AND** the original structural metadata (cell dimensions, atomic numbers, periodicity flags) must remain completely unaltered.

**GIVEN** the `TieredOracle` attempts to route a batch of 500 atomic configurations
**WHEN** precisely 499 structures fall below the MACE uncertainty threshold
**THEN** the `DFTManager` must be invoked exactly once with a payload containing only the 1 singular outlier structure
**AND** the final unified results list must contain all 500 structures, accurately annotated with energies and forces.
