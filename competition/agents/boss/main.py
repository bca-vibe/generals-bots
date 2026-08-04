"""Read generals.bot protocol frames and emit Boss policy actions."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from agent import Agent


@dataclass
class Observation:
    H: int
    W: int
    turn: int
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int
    type_grid: list[list[int]]
    owner_grid: list[list[int]]
    army_grid: list[list[int]]


def _read_grid(stdin, height):
    return [[int(value) for value in stdin.readline().split()] for _ in range(height)]


def main():
    stdin = sys.stdin
    stdout = sys.stdout

    handshake = stdin.readline()
    if not handshake:
        return
    player_id, height, width = (int(value) for value in handshake.split())
    agent = Agent(player_id=player_id, H=height, W=width)

    while True:
        scalars = stdin.readline()
        if not scalars:
            return
        turn, my_land, my_army, opp_land, opp_army = (int(value) for value in scalars.split())
        observation = Observation(
            H=height,
            W=width,
            turn=turn,
            my_land=my_land,
            my_army=my_army,
            opp_land=opp_land,
            opp_army=opp_army,
            type_grid=_read_grid(stdin, height),
            owner_grid=_read_grid(stdin, height),
            army_grid=_read_grid(stdin, height),
        )
        action = agent.act(observation)
        stdout.write("{} {} {} {} {}\n".format(*action))
        stdout.flush()


if __name__ == "__main__":
    main()
