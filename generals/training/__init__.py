"""Training components for the competition policy."""

from .actions import ACTION_COUNT, decode_action, encode_action, legal_action_mask
from .config import TrainingConfig
from .conv_model import (
    ConvCompetitionTransformer,
    ConvPatchResidual,
    calibrate_conv_token_rms,
)
from .model import CompetitionTransformer
from .observation import (
    COMPETITION_OBSERVATION_SCHEMA,
    LEGACY_OBSERVATION_SCHEMA,
    ObservationMemory,
    augment_observation,
    init_observation_memory,
    observation_channel_count,
)

__all__ = [
    "ACTION_COUNT",
    "COMPETITION_OBSERVATION_SCHEMA",
    "CompetitionTransformer",
    "ConvCompetitionTransformer",
    "ConvPatchResidual",
    "LEGACY_OBSERVATION_SCHEMA",
    "ObservationMemory",
    "TrainingConfig",
    "augment_observation",
    "calibrate_conv_token_rms",
    "decode_action",
    "encode_action",
    "init_observation_memory",
    "legal_action_mask",
    "observation_channel_count",
]
