"""Focused behavior and legality tests for the evaluation heuristic agents."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from generals.agents import (
    BossAgent,
    CastleEconomistAgent,
    DeathtouchClockAgent,
    DrawGrinderAgent,
    FogScoutAgent,
    HarvesterAgent,
    HumanExeAgent,
    RaiderAgent,
    SentinelAgent,
)
from generals.agents.heuristic_utils import build_cost_grid
from generals.core.action import DIRECTIONS
from generals.core.observation import Observation


def make_observation(
    *,
    owned_armies=None,
    opponent_armies=None,
    generals=((3, 1), (3, 5)),
    castles=(),
    mountains=(),
    fog=(),
    structures_in_fog=(),
    timestep=0,
):
    shape = (7, 7)
    owned_armies = {(3, 1): 20} if owned_armies is None else owned_armies
    opponent_armies = (
        {(3, 5): 10} if opponent_armies is None else opponent_armies
    )

    armies = jnp.zeros(shape, dtype=jnp.int32)
    owned = jnp.zeros(shape, dtype=jnp.bool_)
    opponent = jnp.zeros(shape, dtype=jnp.bool_)
    for cell, army in owned_armies.items():
        armies = armies.at[cell].set(army)
        owned = owned.at[cell].set(True)
    for cell, army in opponent_armies.items():
        armies = armies.at[cell].set(army)
        opponent = opponent.at[cell].set(True)

    def mask(cells):
        result = jnp.zeros(shape, dtype=jnp.bool_)
        for cell in cells:
            result = result.at[cell].set(True)
        return result

    general_mask = mask(generals)
    castle_mask = mask(castles)
    mountain_mask = mask(mountains)
    fog_mask = mask(fog)
    hidden_structure_mask = mask(structures_in_fog)
    neutral = ~(
        owned
        | opponent
        | mountain_mask
        | fog_mask
        | hidden_structure_mask
    )
    return Observation(
        armies=armies,
        generals=general_mask,
        castles=castle_mask,
        mountains=mountain_mask,
        neutral_cells=neutral,
        owned_cells=owned,
        opponent_cells=opponent,
        fog_cells=fog_mask,
        structures_in_fog=hidden_structure_mask,
        owned_land_count=jnp.sum(owned),
        owned_army_count=jnp.sum(jnp.where(owned, armies, 0)),
        opponent_land_count=jnp.sum(opponent),
        opponent_army_count=jnp.sum(jnp.where(opponent, armies, 0)),
        timestep=jnp.int32(timestep),
    )


def assert_legal_action(observation, action):
    kind, row, col, direction, split = map(int, action)
    assert kind in (0, 1, 2)
    assert split in (0, 1)
    if kind == 1:
        return

    height, width = observation.armies.shape
    assert 0 <= row < height
    assert 0 <= col < width
    assert bool(observation.owned_cells[row, col])

    if kind == 2:
        assert not bool(observation.generals[row, col])
        assert not bool(observation.castles[row, col])
        assert int(observation.armies[row, col]) >= int(
            build_cost_grid(observation)[row, col]
        )
        return

    assert int(observation.armies[row, col]) > 1
    assert 0 <= direction < 4
    destination = jnp.array([row, col]) + DIRECTIONS[direction]
    destination_row, destination_col = map(int, destination)
    assert 0 <= destination_row < height
    assert 0 <= destination_col < width
    assert not bool(observation.mountains[destination_row, destination_col])
    assert not bool(
        observation.structures_in_fog[destination_row, destination_col]
    )


@pytest.mark.parametrize(
    "agent",
    [
        BossAgent(),
        HumanExeAgent(),
        CastleEconomistAgent(),
        SentinelAgent(),
        FogScoutAgent(),
        RaiderAgent(),
        DeathtouchClockAgent(),
        DrawGrinderAgent(),
    ],
)
def test_agents_emit_legal_actions(agent):
    observation = make_observation(
        owned_armies={(3, 1): 20, (3, 2): 60},
        opponent_armies={(3, 5): 10},
        fog=((0, 0), (0, 1), (1, 0)),
        mountains=((2, 2),),
        structures_in_fog=((4, 2),),
    )
    action = agent.act(observation, jax.random.PRNGKey(0))
    assert action.shape == (5,)
    assert action.dtype == jnp.int32
    assert_legal_action(observation, action)


def test_castle_economist_builds_when_a_site_can_afford_it():
    observation = make_observation(
        owned_armies={(3, 1): 20, (3, 2): 60},
        opponent_armies={},
        generals=((3, 1),),
    )
    action = CastleEconomistAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action)) == (2, 3, 2, 0, 0)


def test_boss_builds_a_spaced_castle_with_a_post_build_reserve():
    observation = make_observation(
        owned_armies={(3, 1): 20, (3, 6): 60},
        opponent_armies={},
        generals=((3, 1),),
    )
    action = BossAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action)) == (2, 3, 6, 0, 0)


def test_human_exe_builds_a_spaced_castle_with_a_payback_window():
    observation = make_observation(
        owned_armies={(3, 1): 20, (3, 6): 60},
        opponent_armies={},
        generals=((3, 1),),
        timestep=100,
    )
    action = HumanExeAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action)) == (2, 3, 6, 0, 0)


def test_human_exe_uses_deathtouch_even_when_outnumbered():
    observation = make_observation(
        owned_armies={(3, 3): 2},
        opponent_armies={(3, 4): 50},
        generals=((3, 3), (3, 4)),
        timestep=800,
    )
    action = HumanExeAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 3)


def test_human_exe_memory_infers_a_new_fogged_castle_from_plain_history():
    agent = HumanExeAgent()
    memory = agent.initial_memory(7)
    board_mask = jnp.ones((7, 7), dtype=jnp.bool_)
    first = make_observation(
        owned_armies={(3, 1): 20},
        opponent_armies={},
        generals=((3, 1),),
        timestep=0,
    )
    _, memory = agent.act_with_memory(
        first, jax.random.PRNGKey(0), board_mask, memory
    )
    hidden_build = make_observation(
        owned_armies={(3, 1): 20},
        opponent_armies={},
        generals=((3, 1),),
        structures_in_fog=((0, 0),),
        timestep=10,
    )
    _, memory = agent.act_with_memory(
        hidden_build, jax.random.PRNGKey(1), board_mask, memory
    )
    assert bool(memory.known_castles[0, 0])
    assert not bool(memory.known_mountains[0, 0])


def test_boss_takes_an_immediate_general_win_from_its_own_general():
    observation = make_observation(
        owned_armies={(3, 3): 4},
        opponent_armies={(3, 4): 1},
        generals=((3, 3), (3, 4)),
    )
    action = BossAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 3)


def test_boss_consolidates_instead_of_attacking_a_stronger_home_threat():
    observation = make_observation(
        owned_armies={(3, 1): 5, (3, 3): 10},
        opponent_armies={(3, 4): 20},
        generals=((3, 1),),
    )
    action = BossAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 2)


def test_boss_uses_deathtouch_even_when_the_general_is_outnumbered():
    observation = make_observation(
        owned_armies={(3, 3): 2},
        opponent_armies={(3, 4): 50},
        generals=((3, 3), (3, 4)),
        timestep=800,
    )
    action = BossAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 3)


def test_boss_feeds_only_half_of_a_well_garrisoned_general():
    observation = make_observation(
        owned_armies={(3, 1): 50, (3, 3): 10},
        opponent_armies={},
        generals=((3, 1),),
        fog=((2, 4), (3, 4), (4, 4)),
    )
    action = BossAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[1:3])) == (3, 1)
    assert int(action[4]) == 1


def test_fog_scout_splits_large_armies_across_scouting_fronts():
    observation = make_observation(
        owned_armies={(3, 1): 20},
        opponent_armies={},
        generals=((3, 1),),
        fog=((2, 1), (2, 2), (3, 2), (4, 1), (4, 2)),
    )
    action = FogScoutAgent().act(observation, jax.random.PRNGKey(0))
    assert int(action[0]) == 0
    assert int(action[4]) == 1


def test_fog_scout_checks_capture_strength_after_splitting():
    observation = make_observation(
        owned_armies={(3, 3): 4},
        opponent_armies={(3, 4): 2},
        generals=((3, 3), (3, 4)),
        mountains=((2, 3), (3, 2), (4, 3)),
    )
    action = FogScoutAgent().act(observation, jax.random.PRNGKey(0))
    assert int(action[0]) == 1


def test_harvester_builds_a_spaced_castle_and_keeps_a_reserve():
    observation = make_observation(
        owned_armies={(3, 1): 20, (3, 6): 60},
        opponent_armies={},
        generals=((3, 1),),
    )
    action = HarvesterAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action)) == (2, 3, 6, 0, 0)


def test_raider_routes_toward_a_visible_enemy_castle():
    observation = make_observation(
        owned_armies={(3, 1): 20},
        opponent_armies={(3, 3): 8},
        generals=((3, 1),),
        castles=((3, 3),),
    )
    action = RaiderAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 1, 3)


def test_raider_stages_support_instead_of_taking_a_losing_fight():
    observation = make_observation(
        owned_armies={(3, 1): 5, (3, 3): 10, (4, 3): 8},
        opponent_armies={(3, 4): 20},
        generals=((3, 1),),
        castles=((3, 4),),
    )
    action = RaiderAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 4, 3, 0)


def test_raider_allows_an_outnumbered_deathtouch_win():
    observation = make_observation(
        owned_armies={(3, 3): 2},
        opponent_armies={(3, 4): 50},
        generals=((3, 3), (3, 4)),
        timestep=800,
    )
    action = RaiderAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 3)


def test_sentinel_intercepts_a_visible_threat_without_draining_general():
    observation = make_observation(
        owned_armies={(3, 1): 5, (4, 3): 20},
        opponent_armies={(3, 3): 8},
        generals=((3, 1),),
    )
    action = SentinelAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 4, 3, 0)


def test_sentinel_ignores_a_harmless_distant_enemy_and_reinforces_home():
    observation = make_observation(
        owned_armies={(3, 1): 5, (3, 3): 10},
        opponent_armies={(3, 6): 1},
        generals=((3, 1),),
    )
    action = SentinelAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 2)


def test_sentinel_consolidates_instead_of_attacking_a_stronger_threat():
    observation = make_observation(
        owned_armies={(3, 1): 5, (3, 3): 10},
        opponent_armies={(3, 4): 20},
        generals=((3, 1),),
    )
    action = SentinelAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 2)


def test_sentinel_takes_an_immediate_general_win_before_defending():
    observation = make_observation(
        owned_armies={(3, 3): 4},
        opponent_armies={(3, 4): 1},
        generals=((3, 3), (3, 4)),
    )
    action = SentinelAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 3)


def test_deathtouch_clock_attacks_on_the_hunt_phase_even_when_outnumbered():
    observation = make_observation(
        owned_armies={(3, 1): 1, (3, 3): 2},
        opponent_armies={(3, 4): 50},
        generals=((3, 1), (3, 4)),
        timestep=800,
    )
    action = DeathtouchClockAgent().act(
        observation, jax.random.PRNGKey(0)
    )
    assert tuple(map(int, action[:4])) == (0, 3, 3, 3)


def test_draw_grinder_pulls_excess_armies_back_to_its_general():
    observation = make_observation(
        owned_armies={(3, 1): 5, (3, 3): 10},
        opponent_armies={},
        generals=((3, 1),),
    )
    action = DrawGrinderAgent().act(observation, jax.random.PRNGKey(0))
    assert tuple(map(int, action[:4])) == (0, 3, 3, 2)
