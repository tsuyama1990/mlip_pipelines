# CYCLE03 Specification

## Summary
This specification document meticulously details the absolute architectural requirements, stringent design constraints, and comprehensive implementation pathways specifically tailored for CYCLE03: The Self-Healing DFT Oracle of the PyAcemaker project. The paramount objective of this specific developmental phase is to systematically and incrementally construct a highly resilient, enterprise-grade architecture capable of supporting advanced Machine Learning Interatomic Potential generation.

## System Architecture
The underlying system architecture for Cycle 03 involves highly precise, surgical modifications and sophisticated extensions to the core computational engine. The overarching directory structure perfectly reflects a remarkably clean, strictly modular, and deeply hierarchical design paradigm. We will systematically create, carefully modify, and rigorously test specific files to integrate the novel features absolutely seamlessly into the existing framework.

```text
.
├── src/
│   ├── core/
│   │   ├── orchestrator.py
│   │   └── retry.py
│   ├── domain_models/
│   │   └── config.py
│   └── dynamics/
│       └── dynamics_engine.py
```

The exact, non-negotiable code blueprints strictly dictate that all architectural modifications must be purely additive in nature. We enforce single-responsibility principles via modules like `RetryManager` (`retry.py`), isolating complex network resiliency boundaries from the core sequential orchestrator logic.

## Design Architecture
The fundamental design architecture is entirely and resolutely driven by sophisticated, strictly typed Pydantic-based schemas. This critical development cycle officially introduces a suite of highly robust, deeply complex data models specifically engineered to accurately represent entirely new, nuanced domain concepts crucial for the active learning process. Security checks for arbitrary payloads are ruthlessly enforced at this boundary layer (e.g. `.env` whitelists blocking shell injections and strict size bounds checking).

## Implementation Approach
The highly detailed, meticulously structured, step-by-step implementation guide for Cycle 03 emphatically demands a deeply methodical, rigorously test-driven development (TDD) approach. Cycle 03 introduces the Oracle module, tasked with interfacing with Quantum Espresso. This is the most fragile component, as DFT SCF loops frequently fail. The paramount feature is the implementation of 'Self-Healing Logic'. We will code sophisticated routines that automatically detect charge sloshing or non-convergence, automatically adjust the mixing beta and smearing temperature, and iteratively resubmit the calculation until success.

Path validation must be robust, relying on deterministic Canonical Path mappings to mitigate symlink directory traversal attacks, protecting the underlying active learning sandboxes. Furthermore, all massive binary data cloning must implement chunked streaming logic bound to explicitly sized memory budgets (e.g., 8192 bytes) while simultaneously computing cryptographic signatures to halt Out-Of-Memory (OOM) failures natively.

## Test Strategy
The testing approach for Cycle 03 involves feeding the Oracle with an extensive suite of pre-recorded, intentionally failed Quantum Espresso output logs alongside dynamically patched `tmp_path_factory` validation scopes.

The comprehensive Integration Testing Approach specifically designed for Cycle 03 will rigorously and methodically verify the complex, highly nuanced interactions occurring specifically between the newly implemented computational components and the vast, existing legacy system framework.
