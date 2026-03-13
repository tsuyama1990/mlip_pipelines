import pytest

from src.core.interfaces import (
    AbstractDynamics,
    AbstractGenerator,
    AbstractOracle,
    AbstractTrainer,
    AbstractValidator,
)


def test_cannot_instantiate_generator() -> None:
    with pytest.raises(TypeError):
        AbstractGenerator()  # type: ignore[abstract]

def test_cannot_instantiate_oracle() -> None:
    with pytest.raises(TypeError):
        AbstractOracle()  # type: ignore[abstract]

def test_cannot_instantiate_trainer() -> None:
    with pytest.raises(TypeError):
        AbstractTrainer()  # type: ignore[abstract]

def test_cannot_instantiate_dynamics() -> None:
    with pytest.raises(TypeError):
        AbstractDynamics()  # type: ignore[abstract]

def test_cannot_instantiate_validator() -> None:
    with pytest.raises(TypeError):
        AbstractValidator()  # type: ignore[abstract]
