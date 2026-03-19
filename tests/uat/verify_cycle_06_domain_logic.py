from src.domain_models.config import LoopStrategyConfig


def test_cycle_06_logic() -> None:
    config = LoopStrategyConfig(
        use_tiered_oracle=True,
        incremental_update=True,
        replay_buffer_size=500,
        checkpoint_interval=5,
        timeout_seconds=86400,
    )
    assert config.incremental_update is True
    assert config.use_tiered_oracle is True
