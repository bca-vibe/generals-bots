"""Post-training paired-map benchmark against the in-repository heuristic league."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax

from generals.agents import (
    BossAgent,
    CastleEconomistAgent,
    DeathtouchClockAgent,
    DrawGrinderAgent,
    ExpanderAgent,
    FogScoutAgent,
    HarvesterAgent,
    HumanExeAgent,
    HunterAgent,
    RaiderAgent,
    SentinelAgent,
)

from .evaluation import _random_action

OpponentAction = Callable[[jax.Array, object, jax.Array], jax.Array]


@dataclass(frozen=True)
class OpponentPolicy:
    """Functional opponent interface with optional per-match JAX state."""

    initial_memory: Callable[[int], Any]
    step: Callable[[jax.Array, object, jax.Array, Any], tuple[jax.Array, Any]]

_AGENT_FACTORIES = {
    "boss": BossAgent,
    "expander": ExpanderAgent,
    "hunter": HunterAgent,
    "harvester": HarvesterAgent,
    "human_exe": HumanExeAgent,
    "castle_economist": CastleEconomistAgent,
    "deathtouch_clock": DeathtouchClockAgent,
    "draw_grinder": DrawGrinderAgent,
    "fog_scout": FogScoutAgent,
    "raider": RaiderAgent,
    "sentinel": SentinelAgent,
}


def make_opponent_action(name: str) -> OpponentAction:
    """Return a JAX-traceable action function for one configured opponent."""
    if name == "random":
        return _random_action
    try:
        agent = _AGENT_FACTORIES[name]()
    except KeyError as error:
        raise ValueError(f"Unknown league opponent {name!r}") from error

    def action(key, observation, board_mask):
        del board_mask
        return agent.act(observation, key)

    return action


def make_opponent_policy(name: str) -> OpponentPolicy:
    """Return a state-carrying policy; stateless bots use a scalar dummy state."""
    if name == "human_exe":
        agent = HumanExeAgent()

        def step(key, observation, board_mask, memory):
            return agent.act_with_memory(observation, key, board_mask, memory)

        return OpponentPolicy(agent.initial_memory, step)

    action = make_opponent_action(name)

    def initial_memory(board_size: int):
        del board_size
        return jax.numpy.zeros((), dtype=jax.numpy.int32)

    def step(key, observation, board_mask, memory):
        return action(key, observation, board_mask), memory

    return OpponentPolicy(initial_memory, step)


def aggregate_league_results(
    results: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate equally-sized opponent matchups without hiding the macro view."""
    if not results:
        raise ValueError("Cannot aggregate an empty league")
    wins = sum(result["wins"] for result in results.values())
    losses = sum(result["losses"] for result in results.values())
    draws = sum(result["draws"] for result in results.values())
    games = wins + losses + draws
    macro_score = sum(result["score"] for result in results.values()) / len(results)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games,
        "score": (wins + 0.5 * draws) / games,
        "macro_score": macro_score,
    }
