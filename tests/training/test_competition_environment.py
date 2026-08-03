"""Training's final curriculum stage must be the competition environment."""

from pathlib import Path

import jax
import jax.numpy as jnp

from generals.core.env import GeneralsEnv
from generals.training.config import TrainingConfig
from generals.training.observation import (
    COMPETITION_OBSERVATION_SCHEMA,
    LEGACY_OBSERVATION_SCHEMA,
)
from generals.training.train import make_environment

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "generals"
    / "training"
    / "configs"
    / "competition_l7.toml"
)


def test_final_training_stage_matches_competition_map_parameters_and_boards():
    config = TrainingConfig.from_toml(CONFIG)
    assert config.observation_schema == COMPETITION_OBSERVATION_SCHEMA
    assert config.model_architecture == "transformer"
    assert config.input_channels == 39
    training = make_environment(config, config.curriculum[-1], pool_size=1)
    reference = GeneralsEnv(mode="competition", pool_size=1)

    attributes = (
        "min_grid_size",
        "max_grid_size",
        "pad_to",
        "truncation",
        "mountain_density_range",
        "num_castles_range",
        "min_generals_distance",
        "max_generals_distance",
        "castle_val_range",
        "perfect_info",
        "build_castles",
        "deathtouch_turn",
    )
    for name in attributes:
        assert getattr(training, name) == getattr(reference, name), name

    # Parameter equality should imply equality, but pin one complete generated
    # state too so a future change in make_environment cannot bypass the check.
    key = jax.random.PRNGKey(2026)
    training_state = training._make_single_state_fixed(key, 18, 20)
    reference_state = reference._make_single_state_fixed(key, 18, 20)
    for training_field, reference_field in zip(training_state, reference_state):
        assert jnp.array_equal(training_field, reference_field)


def test_toml_without_schema_defaults_to_legacy(tmp_path):
    config_path = tmp_path / "historical.toml"
    config_path.write_text('[training]\nrun_name = "historical"\n', encoding="utf-8")
    config = TrainingConfig.from_toml(config_path)
    assert config.observation_schema == LEGACY_OBSERVATION_SCHEMA
    assert config.model_architecture == "transformer"
    assert config.input_channels == 38


def test_toml_converts_wandb_tags_to_immutable_tuple(tmp_path):
    config_path = tmp_path / "tracked.toml"
    config_path.write_text(
        '[training]\nwandb_project = "generals-bots"\nwandb_tags = ["conv", "branch"]\n',
        encoding="utf-8",
    )
    config = TrainingConfig.from_toml(config_path)
    assert config.wandb_project == "generals-bots"
    assert config.wandb_tags == ("conv", "branch")


def test_archived_checkpoint_config_is_explicitly_legacy():
    archived = CONFIG.parents[3] / "runs" / "smoke_8xh100" / "smoke_8xh100.toml"
    config = TrainingConfig.from_toml(archived)
    assert config.observation_schema == LEGACY_OBSERVATION_SCHEMA
    assert config.model_architecture == "transformer"
    assert config.input_channels == 38
