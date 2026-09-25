import math

import numpy as np
import pytest

from bcpafl import channel
from bcpafl.mobility import World, exit_time


def test_exit_time_analytic():
    # From the centre moving at 10 m/s, a 1000 m disc is left after 100 s.
    assert exit_time(0, 0, 10, 0, 0, 0, 1000) == pytest.approx(100.0)
    assert exit_time(600, 0, -10, 0, 0, 0, 1000) == pytest.approx(160.0)
    assert exit_time(2000, 0, 1, 0, 0, 0, 1000) == 0.0


def test_vehicles_move_stay_in_area_and_hand_over():
    world = World((("A", 0, 0), ("B", 1800, 0)), 1000, 2400, (20, 25), 0.2,
                  np.random.default_rng(0))
    for i in range(20):
        world.spawn(f"V{i}", "A")
    start = {v: (s.x, s.y) for v, s in world.vehicles.items()}
    world.advance_to(300)
    assert all(abs(s.x) <= 2400 and abs(s.y) <= 2400 for s in world.vehicles.values())
    moved = [math.dist(start[v], (s.x, s.y)) for v, s in world.vehicles.items()]
    assert min(moved) > 0
    assert sum(1 for v in world.vehicles if world.associated_rsu(v) != "A") > 0


def test_channel_degrades_with_distance_and_size():
    p = channel.LinkParams()
    assert channel.capacity_bps(100, p) >= channel.capacity_bps(900, p)
    assert channel.frame_success_prob(100, p) > channel.frame_success_prob(990, p)
    assert channel.message_success_prob(2000, 900, p) > channel.message_success_prob(40000, 900, p)
    rng = np.random.default_rng(0)
    near = sum(channel.transmit_unicast(20000, 100, p, rng).delivered for _ in range(200))
    far = sum(channel.transmit_unicast(20000, 1000, p, rng).delivered for _ in range(200))
    assert near > far
