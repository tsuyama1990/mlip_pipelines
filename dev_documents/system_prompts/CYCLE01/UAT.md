# CYCLE01 User Acceptance Testing

## 1. Test Scenarios

**Scenario ID: UAT-C01-01**
**Priority: High**
**Title: Validation of Active Learning Threshold Constraints and Logic**
This scenario ensures that the two-tier uncertainty thresholds defined in the `ActiveLearningThresholds` model correctly reject unphysical parameters (e.g., setting the global halt threshold lower than the local training threshold). This protects the MD engine from entering an infinite validation loop where it halts to request data but refuses to learn from it, which would stall the entire HPC allocation and waste immense computational resources.

**Scenario ID: UAT-C01-02**
**Priority: High**
**Title: Validation of Cluster Cutout Radii Constraints and Geometric Logic**
This scenario verifies that the `CutoutConfig` correctly mandates that the core extraction radius is strictly and measurably smaller than the surrounding buffer radius. This constraint prevents negative buffer volumes, ensures proper surface passivation space, and guarantees that the cluster extracted for the DFT calculation has a physically meaningful protective shell, preventing SCF divergence.

**Scenario ID: UAT-C01-03**
**Priority: Medium**
**Title: Legacy Configuration Backward Compatibility and Safe Defaults**
This scenario ensures that if a user provides a configuration file from Phase 01 (which lacks the `DistillationConfig` or `CutoutConfig` blocks entirely), the system gracefully upgrades the configuration by applying the safe, carefully chosen defaults without crashing or requiring manual intervention from the user.

## 2. Behavior Definitions

**GIVEN** the system configuration parser is initialized and ready to accept user input
**WHEN** the user provides an `ActiveLearningThresholds` mapping where `threshold_call_dft` is 0.01 and `threshold_add_train` is 0.05 (a logically inverted state)
**THEN** the system must immediately raise a Pydantic ValidationError during initialization
**AND** the error message must explicitly state that the global halt threshold must be strictly greater than or equal to the local training addition threshold to prevent infinite loops.

**GIVEN** the system configuration parser is initialized and ready to accept user input
**WHEN** the user provides a `CutoutConfig` mapping where the `core_radius` is 6.0 and the `buffer_radius` is 4.0 (a physically impossible geometry)
**THEN** the system must immediately raise a Pydantic ValidationError during initialization
**AND** the error message must explicitly state that the buffer radius must be strictly greater than the core radius to ensure a valid physical buffer zone exists.

**GIVEN** the system configuration parser is initialized and ready to process a file
**WHEN** the user provides a minimal, legacy `ProjectConfig` YAML file lacking the entirely new `cutout_config` and `distillation_config` sections
**THEN** the system must successfully parse the legacy configuration without raising any exceptions
**AND** it must automatically inject the default `CutoutConfig` values (e.g., `core_radius=4.0`, `buffer_radius=3.0`, `enable_passivation=True`) into the resulting configuration object, ensuring the system can proceed safely.

**Scenario ID: UAT-C01-04**
**Priority: Low**
**Title: Validation of the Distillation Configuration Overrides**
This scenario verifies that the `DistillationConfig` defaults can be explicitly overridden by the user, ensuring the system correctly parses `mace_model_path` changes and updates the `sampling_structures_per_system` field to control the breadth of the initial zero-shot distillation phase accurately.

**Scenario ID: UAT-C01-05**
**Priority: Low**
**Title: Handling of Unexpected Extra Fields in Configuration**
This scenario confirms that the Pydantic parser, configured with `extra='forbid'`, correctly throws a validation error if a user mistakenly includes unsupported keys or misspelled configurations. This strictness is critical to ensure no silent failures occur when a user intends to configure a threshold but makes a typographical error.

## 3. Extended Behavior Verification

**GIVEN** the system configuration parser is initialized and ready to accept user input
**WHEN** the user provides a complete configuration mapping containing an unexpected key named `invalid_threshold_parameter`
**THEN** the system must immediately raise a Pydantic ValidationError during initialization
**AND** the error message must explicitly state that extra fields are forbidden and highlight the exact location of the misspelled or invalid key to aid user debugging.

**GIVEN** a `DistillationConfig` where `sampling_structures_per_system` is set to -100
**WHEN** the configuration is parsed
**THEN** the system must immediately raise a ValidationError
**AND** the error message must state that the sampling structure count must be an integer strictly greater than zero to ensure meaningful exploration occurs.
