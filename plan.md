1. Set up the Python packages `src/domain_models`, `src/core`, `src/generators`, `src/dynamics`, `src/oracles`, `src/trainers`, `src/validators` with `__init__.py`.
2. Implement Pydantic models in `src/domain_models/config.py` and `src/domain_models/dtos.py`. Ensure all missing fields (`potential_path_template`, `data_directory`, `r_md_mc`, `t_schedule`, `n_defects`, `strain_range`) are present.
3. Write unit tests for domain models in `tests/unit/test_domain_models.py`.
4. Implement `src/generators/adaptive_policy.py` containing the `AdaptivePolicy` engine that analyzes Material DNA and predicted properties to output an `ExplorationStrategy`.
5. Implement `src/dynamics/dynamics_engine.py` (LAMMPS integration, watchdog, OTF halt check).
6. Implement `src/oracles/dft_oracle.py` (Periodic Embedding, Quantum ESPRESSO integration, Self-Healing logic).
7. Implement `src/trainers/ace_trainer.py` (Delta learning, Active set optimization, Pacemaker integration).
8. Implement `src/validators/validator.py` (Phonon dispersion, Mechanical stability, Physical stress tests).
9. Implement `src/core/orchestrator.py` (`ActiveLearningOrchestrator` orchestrating the 6 steps).
10. Update `main.py` to use the `ActiveLearningOrchestrator`.
11. Implement `tests/e2e/test_skeleton.py` and `tutorials/UAT_AND_TUTORIAL.py` (Marimo notebook structure).
12. Run local tests, mypy, and ruff checks to ensure compliance and create `dev_documents/test_execution_log.txt`.
