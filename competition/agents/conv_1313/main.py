"""Stdio entry point for the final iteration-1313 convolutional policy."""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class Observation:
    height: int
    width: int
    turn: int
    my_land: int
    my_army: int
    opponent_land: int
    opponent_army: int
    types: list[list[int]]
    owners: list[list[int]]
    armies: list[list[int]]


def _read_grid(stream, height: int) -> list[list[int]]:
    return [[int(value) for value in stream.readline().split()] for _ in range(height)]


def _read_observation(stream, height: int, width: int, first: bytes) -> Observation:
    turn, my_land, my_army, opponent_land, opponent_army = map(int, first.split())
    return Observation(
        height=height,
        width=width,
        turn=turn,
        my_land=my_land,
        my_army=my_army,
        opponent_land=opponent_land,
        opponent_army=opponent_army,
        types=_read_grid(stream, height),
        owners=_read_grid(stream, height),
        armies=_read_grid(stream, height),
    )


def _warmup() -> None:
    from bot import Conv1313Agent

    agent = Conv1313Agent(21, 21)
    agent.warmup()


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--warmup":
        _warmup()
        return

    stream = sys.stdin.buffer
    handshake = stream.readline()
    if not handshake:
        return
    _player_id, height, width = map(int, handshake.split())

    from bot import Conv1313Agent

    agent = Conv1313Agent(height, width)
    while True:
        first = stream.readline()
        if not first:
            return
        observation = _read_observation(stream, height, width, first)
        action = agent.act(observation)
        sys.stdout.write(f"{action[0]} {action[1]} {action[2]} {action[3]} {action[4]}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
