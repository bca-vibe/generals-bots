"""Standalone CPU inference for the final iteration-1313 conv EMA policy."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BOARD_SIZE = 21
HISTORY_SIZE = 7
TEMPORAL_WINDOW = 512
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
PASS_INDEX = 9 * CELL_COUNT
SCALED_CHANNELS = np.array(
    [0, 1, 2, 12, 14, 15, 16, 17, 18, 24, *range(25, 39)],
    dtype=np.int32,
)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BUILD_OFFSETS = tuple(
    (dr, dc, 14 - 2 * (abs(dr) + abs(dc)))
    for dr in range(-6, 7)
    for dc in range(-6, 7)
    if 14 - 2 * (abs(dr) + abs(dc)) > 0
)


def _linear(x, weight, bias):
    return jnp.einsum("oi,...i->...o", weight, x) + bias


def _layer_norm(x, weight, bias):
    original_dtype = x.dtype
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.maximum(0.0, jnp.var(x, axis=-1, keepdims=True))
    normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
    return (weight.astype(jnp.float32) * normalized + bias.astype(jnp.float32)).astype(original_dtype)


def _group_norm(x, weight, bias):
    original_dtype = x.dtype
    x = x.astype(jnp.float32)
    grouped = x.reshape(12, 8, *x.shape[1:])
    axes = tuple(range(1, grouped.ndim))
    mean = jnp.mean(grouped, axis=axes, keepdims=True)
    variance = jnp.maximum(0.0, jnp.var(grouped, axis=axes, keepdims=True))
    normalized = ((grouped - mean) * jax.lax.rsqrt(variance + 1e-5)).reshape(x.shape)
    affine_shape = (weight.shape[0],) + (1,) * (x.ndim - 1)
    return (
        weight.astype(jnp.float32).reshape(affine_shape) * normalized + bias.astype(jnp.float32).reshape(affine_shape)
    ).astype(original_dtype)


def _conv2d(x, weight, stride, padding):
    return jax.lax.conv_general_dilated(
        x[None],
        weight,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )[0]


def _conv_patch_correction(parameters, observation):
    local = _conv2d(observation, parameters["conv.input_conv.weight"], 1, "SAME")
    local = jax.nn.silu(
        _group_norm(
            local,
            parameters["conv.input_norm.weight"],
            parameters["conv.input_norm.bias"],
        )
    )

    residual = _group_norm(
        local,
        parameters["conv.residual_norm_1.weight"],
        parameters["conv.residual_norm_1.bias"],
    )
    residual = _conv2d(
        jax.nn.silu(residual),
        parameters["conv.residual_conv_1.weight"],
        1,
        "SAME",
    )
    residual = _group_norm(
        residual,
        parameters["conv.residual_norm_2.weight"],
        parameters["conv.residual_norm_2.bias"],
    )
    residual = _conv2d(
        jax.nn.silu(residual),
        parameters["conv.residual_conv_2.weight"],
        1,
        "SAME",
    )
    local = local + residual

    local = _group_norm(
        local,
        parameters["conv.downsample_norm.weight"],
        parameters["conv.downsample_norm.bias"],
    )
    local = _conv2d(
        jax.nn.silu(local),
        parameters["conv.downsample_conv.weight"],
        3,
        "VALID",
    )
    tokens = local.reshape(448, 49).T
    tokens = jax.nn.silu(
        jax.vmap(
            _layer_norm,
            in_axes=(0, None, None),
        )(
            tokens,
            parameters["conv.token_norm.weight"],
            parameters["conv.token_norm.bias"],
        )
    )
    return _linear(
        tokens,
        parameters["conv.output_projection.weight"],
        parameters["conv.output_projection.bias"],
    )


def _policy_logits(parameters, observation, temporal_history, legal_mask):
    observation = observation.at[SCALED_CHANNELS].divide(50.0).astype(jnp.bfloat16)
    patches = observation.reshape(39, 7, 3, 7, 3).transpose(1, 3, 0, 2, 4).reshape(49, 351)
    patch_tokens = _linear(
        patches,
        parameters["patch_embedding.weight"],
        parameters["patch_embedding.bias"],
    )
    patch_tokens = patch_tokens + _conv_patch_correction(parameters, observation)

    temporal_history = temporal_history.astype(jnp.bfloat16)
    army = _linear(
        jax.nn.silu(
            _linear(
                temporal_history[0] / 50.0,
                parameters["temporal_encoder.army_in.weight"],
                parameters["temporal_encoder.army_in.bias"],
            )
        ),
        parameters["temporal_encoder.army_out.weight"],
        parameters["temporal_encoder.army_out.bias"],
    )
    land = _linear(
        jax.nn.silu(
            _linear(
                temporal_history[1] / 50.0,
                parameters["temporal_encoder.land_in.weight"],
                parameters["temporal_encoder.land_in.bias"],
            )
        ),
        parameters["temporal_encoder.land_out.weight"],
        parameters["temporal_encoder.land_out.bias"],
    )
    history_tokens = jnp.stack((army, land)) + parameters["temporal_type_embedding"]
    tokens = jnp.concatenate((parameters["value_token"], history_tokens, patch_tokens))
    tokens = tokens + parameters["position_embedding"]

    for index in range(7):
        prefix = f"blocks.{index}."
        normalized = _layer_norm(
            tokens,
            parameters[prefix + "attention_norm.weight"],
            parameters[prefix + "attention_norm.bias"],
        )
        query = _linear(
            normalized,
            parameters[prefix + "attention.query.weight"],
            parameters[prefix + "attention.query.bias"],
        ).reshape(52, 8, 56)
        key = _linear(
            normalized,
            parameters[prefix + "attention.key.weight"],
            parameters[prefix + "attention.key.bias"],
        ).reshape(52, 8, 56)
        value = _linear(
            normalized,
            parameters[prefix + "attention.value.weight"],
            parameters[prefix + "attention.value.bias"],
        ).reshape(52, 8, 56)
        query, key, value = (item.transpose(1, 0, 2) for item in (query, key, value))
        scores = query @ key.transpose(0, 2, 1) / math.sqrt(56)
        attention = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
        attended = (attention @ value).transpose(1, 0, 2).reshape(52, 448)
        tokens = tokens + _linear(
            attended,
            parameters[prefix + "attention.output.weight"],
            parameters[prefix + "attention.output.bias"],
        )

        normalized = _layer_norm(
            tokens,
            parameters[prefix + "feedforward_norm.weight"],
            parameters[prefix + "feedforward_norm.bias"],
        )
        hidden = jax.nn.silu(
            _linear(
                normalized,
                parameters[prefix + "feedforward_in.weight"],
                parameters[prefix + "feedforward_in.bias"],
            )
        )
        tokens = tokens + _linear(
            hidden,
            parameters[prefix + "feedforward_out.weight"],
            parameters[prefix + "feedforward_out.bias"],
        )

    tokens = _layer_norm(
        tokens,
        parameters["output_norm.weight"],
        parameters["output_norm.bias"],
    )
    patch_logits = _linear(
        tokens[3:],
        parameters["spatial_policy_head.weight"],
        parameters["spatial_policy_head.bias"],
    ).astype(jnp.float32)
    spatial_logits = patch_logits.reshape(7, 7, 9, 3, 3).transpose(2, 0, 3, 1, 4).reshape(-1)
    pass_logit = _linear(tokens[0], parameters["pass_head.weight"], parameters["pass_head.bias"]).astype(jnp.float32)
    logits = jnp.concatenate((spatial_logits, pass_logit))
    if "build_kind_head.weight" in parameters:
        build_kind_residual = _linear(
            tokens[0],
            parameters["build_kind_head.weight"],
            parameters["build_kind_head.bias"],
        ).astype(jnp.float32).reshape(())
        logits = logits.at[8 * CELL_COUNT : 9 * CELL_COUNT].add(
            build_kind_residual
        )
    return jnp.where(legal_mask, logits, -1e9)


def _policy_action(parameters, observation, temporal_history, legal_mask):
    return jnp.argmax(_policy_logits(parameters, observation, temporal_history, legal_mask)).astype(jnp.int32)


class Conv1313Agent:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.board_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self.board_mask[:height, :width] = True

        spatial = (BOARD_SIZE, BOARD_SIZE)
        history = (HISTORY_SIZE, *spatial)
        self.own_army_deltas = np.zeros(history, dtype=np.float32)
        self.enemy_army_deltas = np.zeros(history, dtype=np.float32)
        self.previous_own_army = np.zeros(spatial, dtype=np.float32)
        self.previous_enemy_army = np.zeros(spatial, dtype=np.float32)
        self.known_castles = np.zeros(spatial, dtype=bool)
        self.known_generals = np.zeros(spatial, dtype=bool)
        self.known_mountains = np.zeros(spatial, dtype=bool)
        self.ever_plain = np.zeros(spatial, dtype=bool)
        self.ever_seen = np.zeros(spatial, dtype=bool)
        self.ever_seen_enemy = np.zeros(spatial, dtype=bool)
        self.last_seen_enemy_army = np.zeros(spatial, dtype=np.float32)
        self.last_seen_enemy_age = np.zeros(spatial, dtype=np.float32)
        self.opponent_army_history = np.zeros(TEMPORAL_WINDOW, dtype=np.float32)
        self.opponent_land_history = np.zeros(TEMPORAL_WINDOW, dtype=np.float32)

        weights_path = Path(__file__).with_name("weights.npz")
        with np.load(weights_path, allow_pickle=False) as stored:
            self.parameters = {name: jnp.asarray(stored[name]).view(jnp.bfloat16) for name in stored.files}
        self._forward = jax.jit(_policy_action)

    def warmup(self) -> None:
        observation = jnp.zeros((39, BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
        history = jnp.zeros((2, TEMPORAL_WINDOW), dtype=jnp.float32)
        mask = jnp.ones((PASS_INDEX + 1,), dtype=jnp.bool_)
        jax.block_until_ready(self._forward(self.parameters, observation, history, mask))

    @staticmethod
    def _dilate(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1)
        result = np.zeros_like(mask)
        for dr in range(3):
            for dc in range(3):
                result |= padded[dr : dr + BOARD_SIZE, dc : dc + BOARD_SIZE]
        return result

    @staticmethod
    def _build_cost(own_structures: np.ndarray) -> np.ndarray:
        cost = np.full((BOARD_SIZE, BOARD_SIZE), 35, dtype=np.int32)
        padded = np.pad(own_structures.astype(np.int32), 6)
        for dr, dc, surcharge in BUILD_OFFSETS:
            cost += (
                surcharge
                * padded[
                    6 + dr : 6 + dr + BOARD_SIZE,
                    6 + dc : 6 + dc + BOARD_SIZE,
                ]
            )
        return cost

    def _augment(self, observation):
        types = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        owners = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        armies = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        types[: self.height, : self.width] = observation.types
        owners[: self.height, : self.width] = observation.owners
        armies[: self.height, : self.width] = observation.armies

        fog = (types == 0) & self.board_mask
        mountains = (types == 2) & self.board_mask
        castles = (types == 3) & self.board_mask
        generals = (types == 4) & self.board_mask
        structures_in_fog = (types == 5) & self.board_mask
        owned = (owners == 1) & self.board_mask
        opponent = (owners == 2) & self.board_mask
        neutral = (owners == 0) & (types == 1) & self.board_mask

        own_army = armies * owned
        enemy_army = armies * opponent
        own_delta = own_army - self.previous_own_army
        enemy_delta = enemy_army - self.previous_enemy_army
        self.own_army_deltas[1:] = self.own_army_deltas[:-1]
        self.enemy_army_deltas[1:] = self.enemy_army_deltas[:-1]
        self.own_army_deltas[0] = own_delta
        self.enemy_army_deltas[0] = enemy_delta

        padding = ~self.board_mask
        visible = (~fog & ~structures_in_fog & self.board_mask) | padding
        self.ever_seen |= visible
        self.ever_seen_enemy |= self._dilate(opponent)
        plain_now = self.board_mask & ~mountains & ~castles & ~structures_in_fog
        prior_ever_plain = self.ever_plain.copy()
        initial_fogged_mountains = structures_in_fog & ~prior_ever_plain & ~self.known_castles
        self.known_generals |= generals
        self.known_mountains |= mountains | padding | initial_fogged_mountains
        inferred_castles = structures_in_fog & prior_ever_plain & ~self.known_mountains
        self.known_castles |= castles | inferred_castles
        self.ever_plain |= plain_now

        self.last_seen_enemy_army = np.where(opponent, enemy_army, self.last_seen_enemy_army)
        self.last_seen_enemy_age = np.where(opponent, 0.0, self.last_seen_enemy_age + 1.0)
        self.opponent_army_history[:-1] = self.opponent_army_history[1:]
        self.opponent_land_history[:-1] = self.opponent_land_history[1:]
        self.opponent_army_history[-1] = observation.opponent_army
        self.opponent_land_history[-1] = observation.opponent_land

        coordinate = np.arange(BOARD_SIZE, dtype=np.float32) / (BOARD_SIZE - 1)
        x_coord = np.broadcast_to(coordinate[None, :], (BOARD_SIZE, BOARD_SIZE))
        y_coord = np.broadcast_to(coordinate[:, None], (BOARD_SIZE, BOARD_SIZE))
        ones = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        own_structures = (castles | generals) & owned
        build_cost = self._build_cost(own_structures).astype(np.float32)
        active = (observation.turn >= 800) * ones
        countdown = np.clip((800.0 - observation.turn) / 200.0, 0.0, 1.0) * ones
        channels = np.stack(
            (
                armies,
                own_army,
                enemy_army,
                self.ever_seen,
                self.ever_seen_enemy,
                self.known_generals,
                self.known_castles,
                self.known_mountains,
                neutral,
                owned,
                opponent,
                fog,
                observation.turn * ones,
                (observation.turn % 50) * ones / 50.0,
                observation.my_land * ones,
                observation.my_army * ones,
                observation.opponent_land * ones,
                observation.opponent_army * ones,
                self.last_seen_enemy_army,
                np.log1p(self.last_seen_enemy_age) / 5.0,
                x_coord,
                y_coord,
                active,
                countdown,
                build_cost,
            ),
            dtype=np.float32,
        )
        augmented = np.concatenate((channels, self.own_army_deltas, self.enemy_army_deltas), axis=0)
        temporal = np.stack((self.opponent_army_history, self.opponent_land_history), axis=0)

        self.previous_own_army = own_army
        self.previous_enemy_army = enemy_army
        return augmented, temporal, armies, owned, mountains, castles, generals

    def _legal_mask(self, armies, owned, mountains, castles, generals):
        movement = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        can_move = owned & (armies > 1)
        blocked = mountains | ~self.board_mask
        for direction, (dr, dc) in enumerate(DIRECTIONS):
            for row in range(self.height):
                destination_row = row + dr
                if not 0 <= destination_row < self.height:
                    continue
                for col in range(self.width):
                    destination_col = col + dc
                    if (
                        0 <= destination_col < self.width
                        and can_move[row, col]
                        and not blocked[destination_row, destination_col]
                    ):
                        movement[direction, row, col] = True

        own_structures = (castles | generals) & owned
        build_cost = self._build_cost(own_structures)
        build = owned & ~castles & ~generals & (armies >= build_cost)
        return np.concatenate((movement.reshape(-1), movement.reshape(-1), build.reshape(-1), [True]))

    @staticmethod
    def _decode(index: int) -> tuple[int, int, int, int, int]:
        if index == PASS_INDEX:
            return (1, 0, 0, 0, 0)
        plane, position = divmod(index, CELL_COUNT)
        row, col = divmod(position, BOARD_SIZE)
        if plane == 8:
            return (2, row, col, 0, 0)
        return (0, row, col, plane % 4, int(plane >= 4))

    def act(self, observation) -> tuple[int, int, int, int, int]:
        augmented = self._augment(observation)
        spatial, temporal = augmented[:2]
        legal_mask = self._legal_mask(*augmented[2:])
        action_index = self._forward(
            self.parameters,
            jnp.asarray(spatial),
            jnp.asarray(temporal),
            jnp.asarray(legal_mask),
        )
        return self._decode(int(action_index))
