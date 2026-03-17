# CYCLE01: Frontend Mockups & Core API Scaffolding UAT Plan

## 1. Test Scenarios

### Scenario ID: UAT-01-01: Intent-Driven Translation Validation
**Priority:** High
**Description:** Verify that the system correctly translates a high-level user intent (the "Accuracy vs. Speed" slider) into the complex, deeply nested parameters required by the `ProjectConfig`. This is the core mechanism allowing non-experts to configure the system without knowing hyperparameter syntax. The UAT will execute a pure Python script (via Marimo) that constructs a JSON payload simulating the frontend's output, submits it to the FastAPI endpoint, and verifies the generated configuration. This ensures that the user's abstract request is properly digitized into the exact floating-point tolerances that govern the underlying active learning algorithm's behavior. If this translation is flawed, the entire premise of the intent-driven GUI fails, leading to either glacially slow simulations or wildly inaccurate machine learning potentials.

### Scenario ID: UAT-01-02: Strict Security Validation of GUI Payloads
**Priority:** Critical
**Description:** Verify that the backend strictly rejects any malicious or malformed input originating from the GUI, specifically testing for path traversal injection attempts in string fields and out-of-bounds integers for the trade-off slider. The system must fail gracefully with a descriptive HTTP 422 error, completely protecting the underlying orchestrator. Because the FastAPI backend will likely be exposed on an organizational network or a cloud endpoint, it is absolutely critical that arbitrary string inputs cannot be used to read or overwrite critical files on the host High-Performance Computing cluster. This scenario confirms the integrity of the Pydantic security layer.

### Scenario ID: UAT-01-03: Backward Compatibility with CLI Workflows
**Priority:** Medium
**Description:** Verify that the addition of the new optional `WorkflowIntentConfig` does not break the existing configuration flow for expert users who prefer to submit fully populated YAML files or use the traditional Command Line Interface. The system must natively accept a standard `ProjectConfig` that omits the GUI intent completely and run normally. This ensures that the introduction of the new graphical wrapper does not alienate advanced researchers who rely on highly specific, custom-tuned hyperparameter configurations that fall outside the bounds of the simplified slider abstractions. The system must seamlessly support both user personas simultaneously.

## 2. Behavior Definitions

### UAT-01-01: Intent-Driven Translation Validation
**GIVEN** a running instance of the Adaptive-MLIP FastAPI backend listening on the local interface
**AND** a simulated GUI JSON payload containing a valid `WorkflowIntentConfig` with `accuracy_speed_tradeoff: 1` (representing Maximum Speed)
**WHEN** the payload is submitted via HTTP POST to the `/config/submit` endpoint
**THEN** the system immediately responds with an HTTP 200 OK status code
**AND** the returned serialized JSON configuration shows `distillation_config.uncertainty_threshold` automatically set to a high, permissive value (e.g., 0.15)
**AND** `loop_strategy.replay_buffer_size` is automatically set to a low value to conserve memory
**AND** no syntax or type errors are present anywhere in the final validated configuration tree.

### UAT-01-02: Strict Security Validation of GUI Payloads
**GIVEN** a running instance of the Adaptive-MLIP FastAPI backend
**AND** a simulated GUI JSON payload containing a deliberately malicious path traversal string in the `target_material` field (e.g., `"../../etc/passwd"`)
**WHEN** the payload is submitted via HTTP POST to the `/config/submit` endpoint
**THEN** the system immediately rejects the payload before any internal processing occurs
**AND** responds with an HTTP 422 Unprocessable Entity status code
**AND** the JSON error details explicitly indicate that path traversal sequences or unauthorized characters are forbidden
**AND** the core python orchestrator is never instantiated or invoked in any capacity.

### UAT-01-03: Backward Compatibility with CLI Workflows
**GIVEN** a valid, traditional `ProjectConfig` YAML file that explicitly defines all deep hyperparameters but entirely omits the `WorkflowIntentConfig` or any GUI-specific schema data
**WHEN** the YAML is parsed directly via `ProjectConfig.model_validate_json()` in a standard Python automation script
**THEN** the parsing succeeds immediately without raising any `ValidationError`
**AND** the explicitly defined thresholds and hyperparameters in the YAML are preserved exactly as they were written
**AND** absolutely no mathematical intent translation or parameter overriding occurs, preserving the expert user's exact specifications.
