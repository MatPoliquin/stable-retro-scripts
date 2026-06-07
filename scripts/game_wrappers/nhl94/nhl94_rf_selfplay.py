"""
NHL94 self-play reward functions.
"""

from game_wrappers.nhl94.nhl94_rf import init_attackzone, init_general
from game_wrappers.nhl94.nhl94_rf_defensezone import init_defensezone, isdone_defensezone, rf_defensezone
from game_wrappers.nhl94.nhl94_rf_general import rf_general
from game_wrappers.nhl94.nhl94_rf_scoregoal import isdone_scoregoal, rf_scoregoal


def isdone_selfplay(state):
    t1 = state.team1

    if t1.stats.score > t1.last_stats.score:
        return True

    if state.time < 100:
        return True

    return False


def init_selfplay(env, env_name):
    init_general(env, env_name)


def rf_selfplay(state):
    base = rf_general(state)
    return base


def init_selfplay_offense(env, env_name):
    init_attackzone(env, env_name)


def isdone_selfplay_offense(state):
    return isdone_scoregoal(state)


def rf_selfplay_offense(state):
    return rf_scoregoal(state)


def init_selfplay_defense(env, env_name):
    init_defensezone(env, env_name)


def isdone_selfplay_defense(state):
    return isdone_defensezone(state)


def rf_selfplay_defense(state):
    return rf_defensezone(state)