with open("tests/e2e/test_skeleton.py", "r") as f:
    content = f.read()

content = content.replace(
    "from src.domain_models.config import (\nfrom src.domain_models.config import InterfaceTarget, PolicyConfig, StructureGeneratorConfig\nfrom src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine, FeatureExtractor\nfrom src.generators.structure_generator import StructureGenerator\n\n\n    DynamicsConfig,\n    OracleConfig,\n    ProjectConfig,\n    SystemConfig,\n    TrainerConfig,\n    ValidatorConfig,\n)",
    "from src.domain_models.config import (\n    DynamicsConfig,\n    OracleConfig,\n    ProjectConfig,\n    SystemConfig,\n    TrainerConfig,\n    ValidatorConfig,\n    InterfaceTarget,\n    PolicyConfig,\n    StructureGeneratorConfig\n)\nfrom src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine, FeatureExtractor\nfrom src.generators.structure_generator import StructureGenerator",
)

with open("tests/e2e/test_skeleton.py", "w") as f:
    f.write(content)
