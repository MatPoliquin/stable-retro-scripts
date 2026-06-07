"""
NHL94 keep-puck reward functions.
"""

import math

from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_rf import SampleRandomPos, SetRandomSkaterPositions


def init_keeppuck(env, env_name=None):
    if env_name is None:
        env_name = 'NHL941on1-Genesis-v0'

    SetRandomSkaterPositions(env, env_name, SampleRandomPos)


def isdone_keeppuck(state):
    t1 = state.team1

    if t1.player_haspuck is False:
        return True

    if state.puck.y < 100:
        return True


def rf_keeppuck(state):
    t1 = state.team1

    rew = 1.0
    if not t1.player_haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0
    player = t1.players[controlled_idx]

    speed = math.sqrt(player.vx ** 2 + player.vy ** 2)
    speed_norm = min(1.0, speed / GameConsts.MAX_VEL_XY)

    low_speed_threshold = 0.15
    stagnation_penalty = 0.5 if speed_norm < low_speed_threshold else 0.0

    if t1.player_haspuck and state.puck.y >= 100:
        rew -= stagnation_penalty

    if state.puck.x >= 125 or state.puck.x <= -125:
        rew = -0.5

    return rew