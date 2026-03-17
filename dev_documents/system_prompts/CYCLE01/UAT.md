# CYCLE01: Frontend Mockups & Core API Scaffolding UAT Plan

## 1. Test Scenarios

### Scenario ID: UAT-01-01: Intent-Driven Mathematical Translation Validation
**Priority:** High
**Description:** Verify conclusively that the system correctly, instantly translates a high-level user intent (specifically the "Accuracy vs. Speed" slider parameter) into the complex, deeply nested floating-point parameters implicitly required by the massive `ProjectConfig` engine. This is the absolute core mechanism fundamentally allowing non-experts to perfectly configure the complex system without ever knowing the internal hyperparameter syntax. The automated UAT will flawlessly execute a pure Python script (utilizing the interactive Marimo framework) that programmatically constructs a JSON payload perfectly simulating the massive frontend's output, securely submits it to the FastAPI REST endpoint, and rigorously verifies the generated, translated configuration mathematically. This perfectly ensures that the user's abstract intent is properly digitized into the exact tolerances that govern the incredibly sensitive underlying active learning algorithm's behavior. If this crucial mathematical translation is flawed, the entire premise of the intent-driven GUI fails, inevitably leading to either glacially slow simulations or wildly inaccurate, useless machine learning potentials.

### Scenario ID: UAT-01-02: Computational Budget Safety Override Execution
**Priority:** Critical
**Description:** Verify perfectly that the system strictly and securely enforces the user's selected "Computational Budget" (Low, Medium, High). If a user specifically selects a "LOW" budget but aggressively pushes the "Accuracy" slider to the absolute maximum setting (10), the system's Pydantic validation layer must seamlessly intercept this mathematical conflict. It must decisively override the standard accuracy configuration and firmly cap the maximum number of continuous iterations or active timeout lengths allowed by the `LoopStrategyConfig`. This acts as an absolute fail-safe, decisively proving that the system mathematically protects the user from generating accidental infinite active learning loops that would inevitably financially bankrupt their cloud computing allocations.

### Scenario ID: UAT-01-03: Strict Security Validation of Malicious GUI Payloads
**Priority:** Critical
**Description:** Verify conclusively that the backend server strictly, aggressively rejects any malicious, corrupted, or malformed input originating from the open GUI, specifically rigorously testing for malicious path traversal shell injection attempts heavily embedded within string fields (like `target_material`), and firmly testing for wildly out-of-bounds integer assignments directed at the mathematical trade-off slider. The backend system must fail gracefully and instantly, returning a highly descriptive, sanitized HTTP 422 error, completely protecting the vulnerable, underlying C++ orchestrator binaries. Because the FastAPI web backend will likely be continuously exposed on an internal organizational network or a public cloud endpoint, it is absolutely critical that arbitrary string inputs cannot ever be used to read or overwrite critical execution files on the host High-Performance Computing cluster. This specific scenario definitively confirms the absolute integrity of the Pydantic security validation layer.

### Scenario ID: UAT-01-04: Perfect Backward Compatibility with Legacy CLI Workflows
**Priority:** Medium
**Description:** Verify definitively that the architectural addition of the completely new, optional `WorkflowIntentConfig` schema does absolutely not break or interrupt the existing configuration pipeline flow for expert users and automated systems who prefer to submit fully populated YAML files or utilize the traditional Python Command Line Interface directly. The backend validation system must smoothly, natively accept a standard, traditional `ProjectConfig` object that entirely omits the GUI intent completely and run entirely normally. This ensures that the massive introduction of the new graphical wrapper does not suddenly alienate advanced researchers who rely deeply on highly specific, custom-tuned numerical hyperparameter configurations that purposely fall outside the safe bounds of the simplified graphical slider abstractions. The architecture must flawlessly and seamlessly support both user personas simultaneously without any code branching conflicts.

## 2. Behavior Definitions

### UAT-01-01: Intent-Driven Mathematical Translation Validation
**GIVEN** a running, perfectly healthy instance of the Adaptive-MLIP FastAPI backend listening quietly on the local loopback interface
**AND** a simulated GUI JSON payload containing a flawless, valid `WorkflowIntentConfig` with `accuracy_speed_tradeoff: 1` (securely representing Maximum Speed settings)
**WHEN** the massive JSON payload is actively submitted via HTTP POST to the designated `/config/submit` API endpoint
**THEN** the complex backend system immediately and seamlessly responds with an HTTP 200 OK status code indicating success
**AND** the returned massively serialized JSON configuration clearly shows that `distillation_config.uncertainty_threshold` was automatically set to a high, incredibly permissive floating-point value (e.g., precisely 0.15)
**AND** the deeply nested `loop_strategy.replay_buffer_size` is automatically set to an aggressively low integer value specifically to conserve host memory
**AND** absolutely no syntax or type errors are present anywhere within the massive, final validated configuration tree structure.

### UAT-01-02: Computational Budget Safety Override Execution
**GIVEN** a running instance of the highly secure Adaptive-MLIP FastAPI backend
**AND** a simulated GUI JSON payload containing a conflicting `WorkflowIntentConfig` where `accuracy_speed_tradeoff: 10` (Maximum Accuracy) but simultaneously setting `computational_budget: LOW`
**WHEN** the conflicting payload is explicitly submitted via HTTP POST directly to the `/config/submit` endpoint
**THEN** the validation system intercepts the conflict and responds securely with an HTTP 200 OK status code containing the resolved payload
**AND** the returned validated configuration shows that despite the high accuracy request, the `loop_strategy.max_iterations` attribute is decisively hard-capped to a low numerical limit (e.g., exactly 50 iterations maximum)
**AND** the mathematical rule explicitly proves that the financial limit strictly overrides the theoretical accuracy demands.

### UAT-01-03: Strict Security Validation of Malicious GUI Payloads
**GIVEN** a running, exposed instance of the Adaptive-MLIP FastAPI backend web server
**AND** a simulated GUI JSON payload containing a deliberately malicious, highly crafted path traversal string secretly embedded within the `target_material` field (e.g., passing `"../../etc/shadow"`)
**WHEN** the malicious payload is aggressively submitted via HTTP POST to the vulnerable `/config/submit` endpoint
**THEN** the deeply nested system Pydantic validation tree immediately rejects the payload completely before any internal physics processing or file parsing ever occurs
**AND** the server responds perfectly with an HTTP 422 Unprocessable Entity status code
**AND** the JSON error details explicitly and clearly indicate that path traversal sequences or unauthorized regex characters are strictly forbidden by security rules
**AND** the core python physics orchestrator is absolutely never instantiated, invoked, or touched in any capacity.

### UAT-01-04: Perfect Backward Compatibility with Legacy CLI Workflows
**GIVEN** a completely valid, highly complex traditional `ProjectConfig` YAML file that explicitly defines absolutely all deep hyperparameters manually but entirely omits the `WorkflowIntentConfig` block or any GUI-specific schema data
**WHEN** the massive YAML file is parsed directly and synchronously via `ProjectConfig.model_validate_json()` running within a standard Python automation testing script
**THEN** the complex validation parsing succeeds immediately without raising any internal `ValidationError` exceptions
**AND** the explicitly defined float thresholds and hyperparameters natively residing in the YAML file are preserved exactly and perfectly as they were manually written by the expert
**AND** absolutely no mathematical intent translation, slider mapping, or parameter overriding occurs, beautifully preserving the expert user's exact and precise specifications.
