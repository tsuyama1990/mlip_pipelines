# CYCLE01 Specification

## 1. Summary

CYCLE01 marks the foundational phase of the NextGen Hierarchical Distillation Architecture for the PYACEMAKER project. The primary goal of this cycle is to establish the core data models, rigorous configuration schemas, and strict physical constraints required to govern the advanced Active Learning pipeline safely and deterministically. The success of the entire autonomous system hinges on the absolute reliability of these initial configuration definitions, as they dictate the behavior of every subsequent downstream module, from structure generation to neural network fine-tuning.

In this cycle, we will introduce critical new Pydantic models—specifically `DistillationConfig`, `ActiveLearningThresholds`, `CutoutConfig`, `ExplorationPolicyConfig`, and `LoopStrategyConfig`—and integrate them seamlessly into the existing `ProjectConfig` hierarchy located in `src/domain_models/config.py`. By defining these schemas rigorously using Pydantic, we ensure that all downstream components (Generators, Oracles, Trainers, and Dynamics Engines) interact through strongly-typed, fully validated data transfer objects. This cycle serves as the single source of truth for all pipeline parameters, dictating the operational boundaries for the two-tier thresholding logic, the intelligent cluster extraction geometries, and the foundational model integration pathways.

Furthermore, we must ensure these new structures safely extend the existing architecture without breaking backward compatibility for legacy users. This means implementing intelligent fallback mechanisms and safe defaults. For instance, if a user provides an older configuration file lacking the `CutoutConfig` block, the system must automatically inject a physically sensible default configuration (e.g., a core radius of 3.0 Å and a buffer radius of 5.0 Å) rather than crashing ungracefully. This focus on schema-first development guarantees that the orchestrator never has to deal with missing keys or invalid data types, allowing the core logic to remain clean, concise, and focused entirely on the physics of the simulation rather than mundane data validation tasks. The implementation of this cycle sets the stage for a truly zero-configuration experience, where complex active learning loops can be initiated with absolute confidence in the underlying parameters.

## 2. System Architecture

The architectural goal of this cycle is to solidly establish the Domain Models layer as the unbreakable API contract within the application. The system architecture relies fundamentally on these models acting as the strict interface contract between the central Python Orchestrator and the myriad of executing domain modules (such as LAMMPS wrappers, ASE calculators, and PyTorch foundation models). Without a rigidly defined domain model layer, the system risks passing unvalidated dictionaries across module boundaries, leading to unpredictable runtime crashes during expensive High-Performance Computing (HPC) allocations.

The `ProjectConfig` acts as the root node of the configuration tree. The new models developed in this cycle will be integrated as nested components within this root object, allowing the end-user to provide a single, unified YAML or JSON file to define their entire experimental campaign. The Orchestrator will parse this file exactly once upon initialization, instantaneously validating the entire hierarchical structure. Once parsed, the Orchestrator will pass the validated, immutable Pydantic objects down the dependency chain to the respective modules. This strict boundary management absolutely prevents any configuration mutation during the active learning loops, guaranteeing deterministic and reproducible execution over weeks of continuous simulation.

The architecture mandates that `src/domain_models/config.py` remains entirely independent of heavy external scientific libraries like ASE, LAMMPS, or PyTorch. It must remain pure Python, utilizing only standard libraries and Pydantic. This architectural decision ensures that the configuration parsing logic can be unit-tested instantaneously without incurring the massive overhead of initializing heavy machine learning frameworks or loading compiled C++ simulation engines. It also ensures that the configuration logic can be easily extracted or utilized by external lightweight graphical user interfaces or web dashboards in the future without dragging in massive computational dependencies.

### File Structure Modification

We will carefully modify existing files within the domain models directory to inject these new models, adhering strictly to the additive mindset to preserve existing functionality.

```text
.
├── src/
│   ├── domain_models/
│   │   ├── __init__.py
│   │   ├── **config.py**               (Extend with DistillationConfig, CutoutConfig, etc.)
│   │   ├── **dtos.py**                 (Add FinetuneMetrics and new DTOs)
```

The `dtos.py` file will also be expanded in this cycle to include Data Transfer Objects (DTOs) that facilitate communication between modules during the active learning loop, such as a `CutoutResult` object that encapsulates the extracted `ase.Atoms` object alongside metadata regarding the passivation atoms added.

## 3. Design Architecture

The system is fully designed by Pydantic-based schemas, enforcing strict type safety, automatic validation, and clear conceptual boundaries between the physics configurations and the software execution parameters. This design architecture is fundamental to preventing the "garbage in, garbage out" paradigm that plagues many scientific software projects. By pushing all validation to the outermost boundary of the application, we guarantee that the inner physics engines only ever operate on physically sensible and logically coherent data.

### Domain Concepts and Constraints

1. **`DistillationConfig`:** This schema dictates the parameters for Phase 1 (Zero-Shot Distillation). It captures the choice of the foundation model (e.g., `mace_model_path: str`), the activation flag (`enable: bool`), and the sampling density (`sampling_structures_per_system: int`). The key invariant here is the `uncertainty_threshold` (float), which defines the strict boundary separating MACE's confident predictions from uncertain ones requiring DFT refinement. This value must be strictly positive and typically bounded between 0.0 and 1.0 depending on the underlying variance metric.
2. **`ActiveLearningThresholds`:** This is the core control mechanism for Phase 3. It defines the sophisticated FLARE-inspired two-tier evaluation system. `threshold_call_dft` dictates when the global molecular dynamics simulation should halt and request oracle intervention. `threshold_add_train` defines the stricter, local criterion determining which specific atoms within a clustered cutout are actually added to the machine learning training set. A crucial constraint that must be enforced via Pydantic model validation is `threshold_call_dft >= threshold_add_train`. If this is violated, the system could enter an infinite loop where it halts to call DFT, but then decides none of the atoms are informative enough to learn from. `smooth_steps` prevents thermal noise from causing false-positive halts and must be an integer greater than zero.
3. **`CutoutConfig`:** Governs the intelligent extraction process. `core_radius` and `buffer_radius` represent the spatial boundaries of the spherical cutout. The invariant `core_radius < buffer_radius` must be strictly enforced via Pydantic validators to ensure a physically meaningful buffer volume exists. This model also toggles the automated pre-relaxation and passivation mechanisms via boolean flags.
4. **`LoopStrategyConfig`:** Serves as the high-level policy document for the trainer and orchestrator. It manages the integration of the Tiered Oracle and the `replay_buffer_size`, which is an integer that prevents catastrophic forgetting during incremental ACE updates.

These schemas act as both the definitive documentation and the impassable runtime validation gates for the system, ensuring that user errors are caught immediately with clear, descriptive exception messages rather than manifesting as obscure physics errors deep within a Fortran or C++ binary hours later.

## 4. Implementation Approach

The implementation will strictly follow a schema-first development approach. We will define the data structures and their constraints completely before writing any execution logic in subsequent cycles. This ensures a rock-solid foundation.

1. **Model Definition:** Open `src/domain_models/config.py`. We will define the new classes `DistillationConfig`, `ActiveLearningThresholds`, `CutoutConfig`, `ExplorationPolicyConfig`, and `LoopStrategyConfig`, all strictly inheriting from `pydantic.BaseModel`. We will ensure that `model_config = ConfigDict(extra='forbid')` is set to prevent users from accidentally passing misspelled configuration keys that would otherwise be silently ignored.
2. **Typing and Defaulting:** We will use `pydantic.Field` extensively to set comprehensive descriptions and perfectly safe default values. For instance, `smooth_steps` should default to 3, providing a reasonable buffer against thermal fluctuations, and `passivation_element` should default to `"H"`, which is the most common and robust choice for terminating dangling oxygen or metallic bonds.
3. **Custom Validators:** We will implement `@model_validator(mode='after')` methods for cross-field validation. For `ActiveLearningThresholds`, we will verify that `self.threshold_call_dft >= self.threshold_add_train`. For `CutoutConfig`, we will verify that `self.core_radius < self.buffer_radius` and that both are strictly positive floats. If these constraints are violated, we will raise an explicit, highly descriptive `ValueError` that informs the user exactly how to correct their configuration file.
4. **Integration:** We will locate the main `ProjectConfig` class. We will add the newly defined models as optional fields (e.g., `cutout_config: CutoutConfig = Field(default_factory=CutoutConfig)`). To maintain absolute backward compatibility, we will ensure that if these new sections are missing from a legacy YAML configuration file, Pydantic instantiates them using their default factories seamlessly.
5. **DTO Expansion:** Open `src/domain_models/dtos.py`. We will add necessary Data Transfer Objects to facilitate passing validation metrics or cutout metadata between modules without tightly coupling them. For example, a `ValidationScore` object that encapsulates the RMSE values and Born stability booleans.
6. **Linting and Type Checking:** We will rigorously run `uv run ruff check .` and `uv run mypy .` incrementally during development. We will ensure that the new code strictly adheres to the Ruff formatting rules, specifically avoiding mutable defaults (RUF012) and type hint errors (ANN201). Any necessary exemptions will be carefully considered and documented.

## 5. Test Strategy

Testing will verify the resilience and absolute correctness of the configuration schemas under both ideal and adversarial conditions. We must guarantee that bad configurations cannot slip past the Pydantic boundary.

### Unit Testing Approach
We will exhaustively test the Pydantic boundary conditions within `tests/domain_models/test_config.py`.
- **Valid Instantiation:** We will systematically inject valid Python dictionaries representing the deeply nested structures and assert that the Pydantic models instantiate correctly, maintaining the expected nested types and translating string paths into `pathlib.Path` objects where appropriate.
- **Default Resolution:** We will instantiate the `ProjectConfig` with an entirely empty dictionary or a minimal legacy configuration dictionary. We will assert that the new models (like `CutoutConfig` and `DistillationConfig`) correctly auto-populate with their safe default values, ensuring backward compatibility is flawlessly maintained.
- **Validator Execution:** We will deliberately and maliciously inject invalid data. For example, setting `core_radius=5.0` and `buffer_radius=3.0` in the `CutoutConfig`, or setting `threshold_call_dft=0.01` and `threshold_add_train=0.05` in the `ActiveLearningThresholds`. We will use `pytest.raises(ValueError)` to assert that the custom validators successfully catch these physical impossibilities and raise descriptive, user-friendly error messages. We will also test type coercion failures, such as passing a string to an integer field.

### Integration Testing Approach
While predominantly unit tests, we must ensure these models function correctly in the broader application context, particularly concerning file parsing.
- **File Parsing Integration:** We will write temporary YAML and JSON files representing a complete NextGen configuration using Pytest's `tmp_path` fixture. We will invoke the configuration loading utilities (e.g., `ProjectConfig.model_validate_json()` or standard PyYAML parsing equivalents) and assert that the nested structures are perfectly and losslessly translated from the bytes on disk to strongly typed memory objects.
- **Side-Effect Management:** These tests are entirely self-contained pure functions. We will ensure that no system environment variables or absolute file paths are implicitly required by the configuration loader during testing. We will strictly enforce complete isolation, ensuring that running the test suite does not inadvertently create files in the developer's working directory or attempt to contact external API services.
