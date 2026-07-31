"""Watch a competition-policy checkpoint play a few games against itself.

Games are simulated headlessly and retained only in memory, then opened in the
existing interactive replay GUI. No replay files are written.

    uv run python examples/checkpoint_selfplay.py
    uv run python examples/checkpoint_selfplay.py --mode stochastic --games-per-mode 1

Replay controls: SPACE play/pause | Left/Right or H/L step | R restart | Q next game
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from generals.core import game
from generals.core.observation import Observation
from generals.gui import ReplayGUI
from generals.gui.properties import GuiMode
from generals.training.actions import decode_action, legal_action_mask
from generals.training.config import TrainingConfig
from generals.training.observation import (
    augment_observation,
    init_observation_memory,
    temporal_input,
)
from generals.training.train import (
    _learning_rate,
    _load_checkpoint_state,
    build_network,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPETITION_DIR = REPO_ROOT / "competition"
if str(COMPETITION_DIR) not in sys.path:
    # matchup.py follows the competition runner's script-style import of protocol.py.
    sys.path.insert(0, str(COMPETITION_DIR))

from matchup import make_board, make_transition  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "generals" / "training" / "configs" / "smoke_8xh100.toml"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "smoke_8xh100" / "checkpoint_001260.eqx"


@dataclass
class RecordedGame:
    states: list[game.GameState]
    infos: list[game.GameInfo]
    seed: int
    mode: str
    result: str


def load_ema_checkpoint(config: TrainingConfig, checkpoint: Path):
    """Load the EMA policy and return it with checkpoint iteration metadata."""
    key = jax.random.PRNGKey(config.seed)
    key, network_key = jax.random.split(key)
    network = build_network(config, network_key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    skeleton = (
        network,
        optimizer_state,
        network,
        jnp.int32(0),
        jnp.int32(0),
        key,
    )
    _, _, ema_network, iteration, stage_index, _ = _load_checkpoint_state(
        checkpoint, skeleton, config
    )
    return ema_network, int(iteration), int(stage_index)


def pad_observation(observation: Observation, board_size: int) -> Observation:
    """Pad an exact competition rectangle to the transformer's square input."""
    height, width = observation.armies.shape
    if height > board_size or width > board_size:
        raise ValueError(
            f"Cannot pad {height}x{width} observation to {board_size}x{board_size}"
        )
    padding = ((0, board_size - height), (0, board_size - width))
    return jax.tree.map(
        lambda value: jnp.pad(value, padding) if value.ndim == 2 else value,
        observation,
    )


def make_action_selector(config: TrainingConfig):
    """Build one compiled two-seat inference step with independent memory."""

    @eqx.filter_jit
    def select_actions(network, observations, board_mask, memory, key, greedy):
        board_masks = jnp.stack([board_mask, board_mask])
        augmented, next_memory = jax.vmap(
            lambda observation, current_memory, mask: augment_observation(
                observation,
                current_memory,
                mask,
                config.observation_schema,
            )
        )(observations, memory, board_masks)
        histories = jax.vmap(temporal_input)(next_memory)
        legal_masks = jax.vmap(legal_action_mask)(observations, board_masks)
        logits = jax.vmap(
            lambda observation, history, mask: network.forward(
                observation, history, mask
            )[0]
        )(augmented, histories, legal_masks)

        split_keys = jax.random.split(key, 3)
        sampled_indices = jax.vmap(jax.random.categorical)(split_keys[1:], logits)
        greedy_indices = jnp.argmax(logits, axis=-1)
        indices = jnp.where(greedy, greedy_indices, sampled_indices).astype(jnp.int32)
        actions = jax.vmap(decode_action)(indices)
        selected_actions_are_legal = jnp.take_along_axis(
            legal_masks, indices[:, None], axis=1
        ).all()
        return actions, next_memory, split_keys[0], selected_actions_are_legal

    return select_actions


def play_one_game(
    network,
    config: TrainingConfig,
    select_actions,
    *,
    seed: int,
    mode: str,
) -> RecordedGame:
    """Run one exact-rectangle competition game and retain frames in memory."""
    from generals import GeneralsEnv

    environment = GeneralsEnv(mode="competition")
    transition = make_transition(environment)
    state = make_board(environment, seed)
    height, width = state.armies.shape
    board_mask = jnp.zeros(
        (config.pad_to, config.pad_to), dtype=jnp.bool_
    ).at[:height, :width].set(True)

    base_memory = init_observation_memory(
        config.pad_to, config.history_size, config.temporal_window
    )
    memory = jax.tree.map(lambda value: jnp.stack([value, value]), base_memory)
    policy_key = jax.random.fold_in(jax.random.PRNGKey(seed), 1_260)
    greedy = jnp.asarray(mode == "greedy")

    states = [state]
    infos = [game.get_info(state)]
    for _ in range(environment.truncation):
        observations = jax.tree.map(
            lambda zero, one: jnp.stack([zero, one]),
            pad_observation(game.get_observation(state, 0), config.pad_to),
            pad_observation(game.get_observation(state, 1), config.pad_to),
        )
        actions, memory, policy_key, actions_are_legal = select_actions(
            network, observations, board_mask, memory, policy_key, greedy
        )
        if not bool(actions_are_legal):
            raise AssertionError("Policy selected an action excluded by its legal mask")

        state, info = transition(state, actions)
        states.append(state)
        infos.append(info)
        if bool(info.is_done):
            break

    winner = int(infos[-1].winner)
    if winner >= 0:
        result = f"player {winner} won"
    elif bool(infos[-1].is_done):
        result = "draw by simultaneous capture/deathtouch"
    else:
        result = f"draw at the {environment.truncation}-turn limit"
    return RecordedGame(states, infos, seed, mode, result)


def show_replay(recording: RecordedGame, fps: int, iteration: int) -> None:
    """Show one recording; convert the GUI's Q/SystemExit into 'next game'."""
    label = f"iteration {iteration} EMA ({recording.mode})"
    gui = ReplayGUI(
        recording.states[0],
        agent_ids=[label, label],
        fps=fps,
        mode=GuiMode.REPLAY,
        start_paused=True,
    )
    try:
        gui.play(recording.states, recording.infos)
    except SystemExit:
        # GUI.tick historically raises SystemExit for Q/window-close. Catching it
        # here lets a batch of examples advance without changing the shared GUI.
        gui.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=1260,
                        help="first map seed; subsequent examples increment it")
    parser.add_argument("--games-per-mode", type=int, default=2)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=("stochastic", "greedy", "both"),
        default="both",
        help="policy action selection used for self-play",
    )
    args = parser.parse_args()
    if args.games_per_mode < 1:
        parser.error("--games-per-mode must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_toml(args.config)
    network, iteration, stage_index = load_ema_checkpoint(config, args.checkpoint)
    print(
        f"Loaded EMA checkpoint {args.checkpoint} "
        f"(iteration {iteration}, curriculum stage {stage_index})"
    )

    modes = ("stochastic", "greedy") if args.mode == "both" else (args.mode,)
    select_actions = make_action_selector(config)
    examples = [
        (mode, args.seed + index * args.games_per_mode + game_index)
        for index, mode in enumerate(modes)
        for game_index in range(args.games_per_mode)
    ]
    for number, (mode, seed) in enumerate(examples, start=1):
        print(f"\n[{number}/{len(examples)}] Simulating {mode} self-play, seed {seed}...")
        recording = play_one_game(
            network,
            config,
            select_actions,
            seed=seed,
            mode=mode,
        )
        print(
            f"Seed {seed}: {recording.result} after "
            f"{len(recording.states) - 1} turns on "
            f"{recording.states[0].armies.shape[0]}x"
            f"{recording.states[0].armies.shape[1]}"
        )
        print("Replay open: SPACE play/pause | arrows or H/L step | R restart | Q next")
        show_replay(recording, args.fps, iteration)

    print("\nFinished all self-play examples.")


if __name__ == "__main__":
    main()
