"""Harvester: Hunter, plus it builds or captures production along the way.

A minimal improvement on the Hunter. Army income is dominated by *structures*:
the general and every owned **castle** grow +1 every 2 turns, while plain land
grows only +1 per 50 turns. Hunter and Expander both route around cities and
never take them. The Harvester runs Hunter's garrison / conveyor / kill machinery,
but captures neutral castles when available and converts large, well-spaced army
piles into new castles under the competition build rules.
"""
import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent
from .heuristic_utils import best_build, build_cost_grid, distance_field
from .hunter_agent import GARRISON, _bfs, _toward

CASTLE_RESERVE = 8
MIN_CASTLE_SPACING = 5


@jax.jit
def harvester_action(key, obs):
    """Capture general > build/bank production > feed surplus > advance > wait."""
    del key
    a, mine = obs.armies, obs.owned_cells
    H, W = a.shape
    reach = jnp.int32(H * W)
    mine_army = jnp.where(mine, a, 0)
    movable = mine & (a > 1)
    biggest = jnp.max(mine_army)

    gen = mine & obs.generals
    gen_army = jnp.sum(jnp.where(gen, a, 0))
    g = jnp.argmax(gen.reshape(-1).astype(jnp.int32))

    # Build only from a pile that remains useful afterward, and keep production
    # spread out so proximity surcharges do not consume the conveyor stack.
    structures = mine & (obs.generals | obs.castles)
    structure_distance = distance_field(jnp.ones_like(mine), structures)
    build_cost = build_cost_grid(obs)
    build_candidate = (
        mine
        & ~obs.generals
        & ~obs.castles
        & (structure_distance >= MIN_CASTLE_SPACING)
        & (a >= build_cost + CASTLE_RESERVE)
    )
    build_scores = (
        (a - build_cost).astype(jnp.float32) * 4.0
        + jnp.minimum(structure_distance, 7).astype(jnp.float32) * 10.0
    )

    # Affordable neutral castles become walkable targets; other cities stay walls.
    affordable_city = obs.cities & obs.neutral_cells & (biggest - 1 > a)
    passable = ~(obs.mountains | obs.structures_in_fog | (obs.cities & ~mine & ~affordable_city))
    from_gen = _bfs(passable, gen)

    # Goal: enemy general > an affordable city to bank > nearest enemy land > scout.
    egen = obs.opponent_cells & obs.generals
    build_action = best_build(
        obs,
        build_scores,
        candidate_mask=build_candidate & ~jnp.any(egen),
    )
    enemy = obs.opponent_cells & ~obs.cities
    fog = obs.fog_cells & passable & (from_gen < reach)
    open_ = passable & ~mine & (from_gen < reach)

    def farthest(mask):
        return mask & (from_gen == jnp.max(jnp.where(mask, from_gen, -1)))

    goal = jnp.where(jnp.any(egen), egen,
           jnp.where(jnp.any(affordable_city), affordable_city,
           jnp.where(jnp.any(enemy), enemy,
           jnp.where(jnp.any(fog), farthest(fog), farthest(open_)))))

    to_goal = _bfs(passable, goal)
    direction, nbr = _toward(to_goal, passable)
    advances = nbr < to_goal
    dirn = direction.reshape(-1)

    egen_army = jnp.sum(jnp.where(egen, a, 0))
    kill = jnp.any(egen) & movable & (to_goal == 1) & advances & (a - 1 > egen_army)
    ki = jnp.argmax(jnp.where(kill, mine_army, -1).reshape(-1))
    feed = (gen_army >= 2 * GARRISON) & advances.reshape(-1)[g]
    fwd = movable & ~gen & advances
    ci = jnp.argmax(jnp.where(fwd, mine_army, -1).reshape(-1))

    do_kill = jnp.any(kill)
    do_build = ~do_kill & (build_action[0] == 2)
    do_feed = ~do_kill & ~do_build & feed
    do_conv = ~do_kill & ~do_build & ~do_feed & jnp.any(fwd)
    i = jnp.where(do_kill, ki, jnp.where(do_feed, g, ci))
    move_action = jnp.array(
        [~(do_kill | do_feed | do_conv), i // W, i % W, dirn[i], do_feed],
        dtype=jnp.int32,
    )
    return jnp.where(do_build, build_action, move_action)


class HarvesterAgent(Agent):
    """Runs Hunter's playbook while building and banking production structures."""

    def __init__(self, id: str = "Harvester"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        return harvester_action(key, observation)

    def reset(self):
        pass
