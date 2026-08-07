"""Typed configuration for baseline self-play training."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .observation import (
    COMPETITION_OBSERVATION_SCHEMA,
    LEGACY_OBSERVATION_SCHEMA,
    OBSERVATION_SCHEMAS,
    observation_channel_count,
)

MODEL_ARCHITECTURES = frozenset(("transformer", "conv_transformer"))
LEAGUE_OPPONENTS = frozenset(
    (
        "random",
        "boss",
        "expander",
        "hunter",
        "harvester",
        "human_exe",
        "castle_economist",
        "deathtouch_clock",
        "draw_grinder",
        "fog_scout",
        "raider",
        "sentinel",
    )
)


@dataclass(frozen=True)
class CurriculumStage:
    min_generals_distance: int
    max_generals_distance: int | None = None
    advance_win_rate: float = 0.60


@dataclass(frozen=True)
class TrainingConfig:
    # Run and output.
    run_name: str = "competition_l7_baseline"
    seed: int = 44
    output_dir: str = "checkpoints"
    # Optional live experiment tracking. Setting WANDB_PROJECT in the
    # environment also enables it without changing a checked-in recipe.
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_run_id: str | None = None
    wandb_run_name: str | None = None
    wandb_job_type: str = "training"
    parent_wandb_run_id: str | None = None
    parent_wandb_url: str | None = None
    parent_final_iteration: int = 0
    parent_final_samples: int = 0
    parent_wall_seconds: float = 0.0
    resume_checkpoint_source: str | None = None
    resume_checkpoint_sha256: str | None = None
    resume_raw_weights: bool = False
    resume_optimizer_state: bool = False
    resume_ema_weights: bool = False
    # True when the resume checkpoint already contains the auxiliary castle
    # difference and residual build-kind heads. Historical parent checkpoints
    # leave this false so continuation can attach zero-initialized heads.
    resume_checkpoint_has_counterfactual_heads: bool = False
    resume_start_stage: int = -1
    previous_device_count: int = 0
    previous_gpu_layout: str | None = None
    current_gpu_layout: str | None = None
    previous_num_envs_per_device: int = 0
    previous_minibatch_size_per_device: int = 0
    preserved_global_envs: int = 0
    preserved_global_minibatch_size: int = 0

    # The competition rules remain fixed throughout the curriculum.
    pad_to: int = 21
    min_grid_size: int = 18
    max_grid_size: int = 21
    truncation: int = 1200
    deathtouch_turn: int = 800
    # These are the official generator's pre-strip settings. It samples 9-11
    # mountain cells as temporary neutral castles, then build-castles mode
    # strips them to plains. The played maps consequently land in the published
    # approximate 19-23% / 65-105 final mountain range.
    mountain_density_min: float = 0.24
    mountain_density_max: float = 0.26
    pool_size: int = 200_000
    reset_pool_every: int = 20

    # AverageJoe L_7d architecture.
    history_size: int = 7
    temporal_window: int = 512
    # Legacy is the safe default for historical TOMLs that predate schema
    # versioning. New training configs opt into competition_39 explicitly.
    observation_schema: str = LEGACY_OBSERVATION_SCHEMA
    model_architecture: str = "transformer"
    depth: int = 7
    embed_dim: int = 448
    attention_heads: int = 8
    ff_factor: int = 3
    patch_size: int = 3
    use_bf16: bool = True
    value_bins: int = 128
    value_min: float = -1.0
    value_max: float = 1.0
    hl_gauss_sigma: float = 0.04
    conv_channels: int = 96
    conv_groups: int = 12
    # Fresh convolutional runs rescale the branch's output projection once,
    # using real initial competition observations, so its token correction has
    # this RMS ratio relative to the ordinary patch-token stream.
    conv_initial_token_rms_ratio: float = 0.10
    conv_calibration_samples: int = 512

    # Each accelerator runs this many environments. Both player seats become
    # training samples, so samples/iteration/device = 2*num_envs*num_steps.
    num_envs: int = 512
    num_steps: int = 512
    num_iterations: int = 100_000
    minibatch_size: int = 1024
    ppo_epochs: int = 1

    # PPO recipe from the released L_7d run.
    gamma: float = 1.0
    gae_lambda: float = 0.90
    clip_epsilon: float = 0.20
    value_coefficient: float = 0.50
    max_grad_norm: float = 0.267
    target_kl: float = 0.02
    advantage_top_fraction: float = 0.25

    # Optional castle-discovery treatment. The critic always targets the raw
    # terminal return; potential shaping affects only the actor's GAE stream.
    actor_potential_shaping: bool = False
    castle_potential_scale: float = 1.0
    tactical_build_logit_boost: float = 0.0
    castle_intervention_full_until: int = 0
    castle_intervention_anneal_until: int = 0
    tactical_build_post_reserve: int = 10
    tactical_build_payback_margin: int = 25
    tactical_build_remembered_enemy_turns: int = 50
    tactical_build_enemy_safety_radius: int = 3

    # Optional supervised castle counterfactual stream. When disabled the
    # historical network/checkpoint tree and PPO update path are unchanged.
    # A zero-initialized residual build-kind head is deployment-visible but
    # distribution preserving at attachment time.
    residual_build_kind_head: bool = False
    counterfactual_castle_training: bool = False
    counterfactual_schema_version: int = 1
    counterfactual_refresh_every: int = 20
    counterfactual_source_states: int = 256
    counterfactual_source_games_per_device: int = 64
    counterfactual_build_candidates: int = 2
    counterfactual_repetitions: int = 4
    counterfactual_buffer_capacity: int = 30_000
    counterfactual_max_age_iterations: int = 100
    counterfactual_actor_minibatch_size_per_device: int = 64
    counterfactual_successor_minibatch_size_per_device: int = 64
    counterfactual_update_every_minibatches: int = 1
    counterfactual_unique_examples_full_weight: int = 2048
    counterfactual_uniform_source_states: int = 128
    counterfactual_promising_source_states: int = 64
    counterfactual_hard_source_states: int = 64
    counterfactual_actor_uniform_per_device: int = 32
    counterfactual_actor_positive_per_device: int = 16
    counterfactual_actor_negative_per_device: int = 16
    counterfactual_recent_fraction: float = 0.75
    counterfactual_actor_coefficient: float = 0.05
    counterfactual_value_coefficient: float = 0.05
    counterfactual_delta_coefficient: float = 0.05
    counterfactual_actor_temperature: float = 0.25
    counterfactual_actor_weight_scale: float = 0.5
    counterfactual_uncertainty_kappa: float = 1.0
    counterfactual_huber_delta: float = 0.25
    counterfactual_rollout_steps: int = 1200
    # Independent global-iteration schedules. A zero pair preserves the
    # historical constant-coefficient behaviour. Otherwise the multiplier is
    # one through ``start`` and linearly reaches zero at ``end``.
    counterfactual_actor_anneal_start: int = 0
    counterfactual_actor_anneal_end: int = 0
    counterfactual_critic_anneal_start: int = 0
    counterfactual_critic_anneal_end: int = 0

    learning_rate_numerator: float = 0.5
    learning_rate_exponent: float = 1.1
    learning_rate_min: float = 5e-6
    learning_rate_max: float = 1e-4
    entropy_start: float = 0.05
    entropy_power: float = 0.2
    entropy_min: float = 0.001

    ema_decay: float = 0.999
    eval_every: int = 50
    eval_games: int = 512
    checkpoint_every: int = 500
    checkpoint_extra_iterations: tuple[int, ...] = ()
    # A replace-in-place recovery checkpoint can be more frequent than the
    # numbered archival checkpoint. Zero preserves the historical behaviour.
    latest_checkpoint_every: int = 0
    metrics_every: int = 1
    # For fixed continuation experiments, schedule eval/checkpoint cadences
    # from the resume boundary rather than absolute global iteration numbers.
    cadence_relative_to_resume: bool = False
    # Optional post-training benchmark. Unlike the random-policy curriculum
    # gate, the league always uses the final competition map-distance stage.
    league_eval_after_training: bool = False
    league_eval_every: int = 0
    league_eval_maps: int = 256
    league_eval_seed: int = 1044
    league_opponents: tuple[str, ...] = ()
    league_eval_policies: tuple[str, ...] = ("ema",)
    league_periodic_raw_max_overhead_fraction: float = 0.10
    # Optional fixed neural opponent, evaluated separately from the heuristic
    # aggregate. The checkpoint is loaded once and kept immutable.
    league_checkpoint_name: str | None = None
    league_checkpoint_path: str | None = None
    league_checkpoint_sha256: str | None = None
    league_checkpoint_maps: int = 128
    league_checkpoint_policy: str = "ema"
    # Growing neural checkpoint league. Raw policies only face raw snapshots;
    # EMA policies only face EMA snapshots. The current checkpoint is added
    # after evaluation on each ``add_every`` boundary.
    learned_league_eval_every: int = 0
    learned_league_add_every: int = 0
    learned_league_extra_eval_iterations: tuple[int, ...] = ()
    learned_league_maps: int = 128
    learned_league_seed: int = 2044
    learned_league_anchor_name: str | None = None
    learned_league_anchor_path: str | None = None
    learned_league_anchor_sha256: str | None = None
    terminal_raw_ema_head_to_head_maps: int = 0
    # Insert device barriers to time rollout/update phases separately. Costs
    # throughput (serializes host glue with device work) - profiling runs only.
    debug_timing: bool = False

    curriculum: tuple[CurriculumStage, ...] = field(
        default_factory=lambda: (
            CurriculumStage(2, 6),
            CurriculumStage(4, 8),
            CurriculumStage(6, 13),
            CurriculumStage(11, 17),
            CurriculumStage(17, None),
        )
    )

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    @property
    def input_channels(self) -> int:
        return observation_channel_count(self.observation_schema, self.history_size)

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainingConfig:
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        data = raw.get("training", raw)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown training config fields: {unknown}")
        if "curriculum" in data:
            data["curriculum"] = tuple(CurriculumStage(**stage) for stage in data["curriculum"])
        if "wandb_tags" in data:
            data["wandb_tags"] = tuple(data["wandb_tags"])
        if "league_opponents" in data:
            data["league_opponents"] = tuple(data["league_opponents"])
        if "league_eval_policies" in data:
            data["league_eval_policies"] = tuple(data["league_eval_policies"])
        if "checkpoint_extra_iterations" in data:
            data["checkpoint_extra_iterations"] = tuple(
                data["checkpoint_extra_iterations"]
            )
        if "learned_league_extra_eval_iterations" in data:
            data["learned_league_extra_eval_iterations"] = tuple(
                data["learned_league_extra_eval_iterations"]
            )
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        for name in (
            "wandb_project",
            "wandb_entity",
            "wandb_group",
            "wandb_run_id",
            "wandb_run_name",
            "parent_wandb_run_id",
            "parent_wandb_url",
            "resume_checkpoint_source",
            "resume_checkpoint_sha256",
            "previous_gpu_layout",
            "current_gpu_layout",
            "league_checkpoint_name",
            "league_checkpoint_path",
            "league_checkpoint_sha256",
            "learned_league_anchor_name",
            "learned_league_anchor_path",
            "learned_league_anchor_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must be non-empty when specified")
        if not self.wandb_job_type.strip():
            raise ValueError("wandb_job_type must be non-empty")
        if (
            self.parent_final_iteration < 0
            or self.parent_final_samples < 0
            or self.parent_wall_seconds < 0
        ):
            raise ValueError("parent iteration and wall time must be non-negative")
        if self.resume_start_stage < -1:
            raise ValueError("resume_start_stage must be -1 or non-negative")
        if any(
            value < 0
            for value in (
                self.previous_device_count,
                self.previous_num_envs_per_device,
                self.previous_minibatch_size_per_device,
                self.preserved_global_envs,
                self.preserved_global_minibatch_size,
            )
        ):
            raise ValueError("lineage batch and device counts must be non-negative")
        for name in (
            "resume_checkpoint_sha256",
            "league_checkpoint_sha256",
            "learned_league_anchor_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if any(not tag.strip() for tag in self.wandb_tags):
            raise ValueError("wandb_tags cannot contain empty values")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.counterfactual_schema_version <= 0:
            raise ValueError("counterfactual_schema_version must be positive")
        positive_counterfactual_ints = (
            "counterfactual_refresh_every",
            "counterfactual_source_states",
            "counterfactual_source_games_per_device",
            "counterfactual_build_candidates",
            "counterfactual_repetitions",
            "counterfactual_buffer_capacity",
            "counterfactual_max_age_iterations",
            "counterfactual_actor_minibatch_size_per_device",
            "counterfactual_successor_minibatch_size_per_device",
            "counterfactual_update_every_minibatches",
            "counterfactual_unique_examples_full_weight",
            "counterfactual_rollout_steps",
        )
        for name in positive_counterfactual_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.counterfactual_build_candidates not in (1, 2):
            raise ValueError("counterfactual_build_candidates must be 1 or 2")
        if self.counterfactual_rollout_steps > self.truncation:
            raise ValueError("counterfactual_rollout_steps cannot exceed truncation")
        source_quota = (
            self.counterfactual_uniform_source_states
            + self.counterfactual_promising_source_states
            + self.counterfactual_hard_source_states
        )
        if source_quota != self.counterfactual_source_states:
            raise ValueError(
                "counterfactual source quotas must sum to counterfactual_source_states"
            )
        actor_quota = (
            self.counterfactual_actor_uniform_per_device
            + self.counterfactual_actor_positive_per_device
            + self.counterfactual_actor_negative_per_device
        )
        if actor_quota != self.counterfactual_actor_minibatch_size_per_device:
            raise ValueError(
                "counterfactual actor quotas must sum to the per-device actor minibatch"
            )
        if not 0.0 <= self.counterfactual_recent_fraction <= 1.0:
            raise ValueError("counterfactual_recent_fraction must be in [0, 1]")
        for name in (
            "counterfactual_actor_coefficient",
            "counterfactual_value_coefficient",
            "counterfactual_delta_coefficient",
            "counterfactual_actor_temperature",
            "counterfactual_actor_weight_scale",
            "counterfactual_uncertainty_kappa",
            "counterfactual_huber_delta",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.counterfactual_castle_training:
            if self.model_architecture != "conv_transformer":
                raise ValueError(
                    "counterfactual castle training currently requires conv_transformer"
                )
            if self.counterfactual_actor_temperature == 0:
                raise ValueError("counterfactual_actor_temperature must be positive")
            if self.counterfactual_actor_weight_scale == 0:
                raise ValueError("counterfactual_actor_weight_scale must be positive")
        for prefix in ("counterfactual_actor", "counterfactual_critic"):
            start = getattr(self, f"{prefix}_anneal_start")
            end = getattr(self, f"{prefix}_anneal_end")
            if start < 0 or end < 0 or end < start:
                raise ValueError(
                    f"{prefix} anneal bounds must be non-negative with end >= start"
                )
            if bool(start) != bool(end):
                raise ValueError(
                    f"{prefix} anneal start and end must both be zero or both positive"
                )
        if self.residual_build_kind_head and self.model_architecture != "conv_transformer":
            raise ValueError(
                "residual_build_kind_head currently requires conv_transformer"
            )
        if (
            self.eval_every < 0
            or self.checkpoint_every <= 0
            or self.latest_checkpoint_every < 0
            or self.metrics_every <= 0
            or self.league_eval_every < 0
            or self.learned_league_eval_every < 0
            or self.learned_league_add_every < 0
            or self.terminal_raw_ema_head_to_head_maps < 0
        ):
            raise ValueError(
                "eval_every, latest_checkpoint_every, and league_eval_every must be "
                "non-negative; checkpoint_every and metrics_every must be positive"
            )
        if self.league_eval_maps <= 0:
            raise ValueError("league_eval_maps must be positive")
        if self.league_checkpoint_maps <= 0:
            raise ValueError("league_checkpoint_maps must be positive")
        if self.learned_league_maps <= 0:
            raise ValueError("learned_league_maps must be positive")
        for name, values in (
            ("checkpoint_extra_iterations", self.checkpoint_extra_iterations),
            (
                "learned_league_extra_eval_iterations",
                self.learned_league_extra_eval_iterations,
            ),
        ):
            if any(value <= 0 for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct positive iterations")
        checkpoint_fields = (
            self.league_checkpoint_name,
            self.league_checkpoint_path,
            self.league_checkpoint_sha256,
        )
        if any(checkpoint_fields) and not all(checkpoint_fields):
            raise ValueError(
                "league checkpoint name, path, and sha256 must be provided together"
            )
        if self.league_checkpoint_policy not in {"raw", "ema"}:
            raise ValueError("league_checkpoint_policy must be 'raw' or 'ema'")
        learned_anchor_fields = (
            self.learned_league_anchor_name,
            self.learned_league_anchor_path,
            self.learned_league_anchor_sha256,
        )
        if any(learned_anchor_fields) and not all(learned_anchor_fields):
            raise ValueError(
                "learned league anchor name, path, and sha256 must be provided together"
            )
        learned_enabled = bool(
            self.learned_league_eval_every
            or self.learned_league_extra_eval_iterations
        )
        if learned_enabled and not all(learned_anchor_fields):
            raise ValueError("learned league evaluation requires an anchor checkpoint")
        if learned_enabled and self.learned_league_add_every <= 0:
            raise ValueError("learned league add_every must be positive when enabled")
        unknown_opponents = sorted(set(self.league_opponents) - LEAGUE_OPPONENTS)
        if unknown_opponents:
            raise ValueError(f"Unknown league opponents: {unknown_opponents}")
        if len(set(self.league_opponents)) != len(self.league_opponents):
            raise ValueError("league_opponents cannot contain duplicates")
        unknown_policies = sorted(set(self.league_eval_policies) - {"raw", "ema"})
        if unknown_policies:
            raise ValueError(f"Unknown league evaluation policies: {unknown_policies}")
        if len(set(self.league_eval_policies)) != len(self.league_eval_policies):
            raise ValueError("league_eval_policies cannot contain duplicates")
        if not 0 <= self.league_periodic_raw_max_overhead_fraction <= 1:
            raise ValueError(
                "league_periodic_raw_max_overhead_fraction must be between zero and one"
            )
        if self.castle_potential_scale < 0 or self.tactical_build_logit_boost < 0:
            raise ValueError("castle shaping scale and build-logit boost must be non-negative")
        if (
            self.castle_intervention_full_until < 0
            or self.castle_intervention_anneal_until < 0
            or self.castle_intervention_anneal_until
            < self.castle_intervention_full_until
        ):
            raise ValueError(
                "castle intervention boundaries must be non-negative and anneal_until "
                "must be at least full_until"
            )
        if any(
            value < 0
            for value in (
                self.tactical_build_post_reserve,
                self.tactical_build_payback_margin,
                self.tactical_build_remembered_enemy_turns,
                self.tactical_build_enemy_safety_radius,
            )
        ):
            raise ValueError("tactical build gate settings must be non-negative")
        league_enabled = bool(self.league_eval_after_training or self.league_eval_every)
        fixed_checkpoint_enabled = bool(self.league_checkpoint_path)
        if (
            league_enabled
            and not self.league_opponents
            and not fixed_checkpoint_enabled
        ):
            raise ValueError(
                "league evaluation requires league_opponents or league_checkpoint_path"
            )
        if league_enabled and not self.league_eval_policies:
            raise ValueError(
                "league_eval_policies must be non-empty when league evaluation is enabled"
            )
        if self.pad_to != 21:
            raise ValueError("The competition baseline is intentionally fixed to pad_to=21")
        if self.observation_schema not in OBSERVATION_SCHEMAS:
            raise ValueError(
                f"observation_schema must be one of {sorted(OBSERVATION_SCHEMAS)}"
            )
        if self.model_architecture not in MODEL_ARCHITECTURES:
            raise ValueError(
                f"model_architecture must be one of {sorted(MODEL_ARCHITECTURES)}"
            )
        if (
            self.model_architecture == "conv_transformer"
            and self.observation_schema != COMPETITION_OBSERVATION_SCHEMA
        ):
            raise ValueError("conv_transformer requires observation_schema='competition_39'")
        if self.conv_channels <= 0 or self.conv_groups <= 0:
            raise ValueError("conv_channels and conv_groups must be positive")
        if self.conv_channels % self.conv_groups:
            raise ValueError("conv_channels must be divisible by conv_groups")
        if not 0 < self.conv_initial_token_rms_ratio < 1:
            raise ValueError(
                "conv_initial_token_rms_ratio must be strictly between zero and one"
            )
        if self.conv_calibration_samples <= 0:
            raise ValueError("conv_calibration_samples must be positive")
        if self.patch_size != 3 or self.pad_to % self.patch_size:
            raise ValueError("pad_to must be divisible by the 3x3 patch size")
        if self.embed_dim % self.attention_heads:
            raise ValueError("embed_dim must be divisible by attention_heads")
        samples = 2 * self.num_envs * self.num_steps
        kept = int(samples * self.advantage_top_fraction)
        if kept < self.minibatch_size:
            raise ValueError("advantage filtering retains fewer than one minibatch")
        if kept % self.minibatch_size:
            raise ValueError(
                "2*num_envs*num_steps*advantage_top_fraction must be divisible by minibatch_size"
            )
        if not self.curriculum:
            raise ValueError("At least one curriculum stage is required")
        if not 0 <= self.mountain_density_min <= self.mountain_density_max <= 1:
            raise ValueError("Mountain-density bounds must satisfy 0 <= min <= max <= 1")
