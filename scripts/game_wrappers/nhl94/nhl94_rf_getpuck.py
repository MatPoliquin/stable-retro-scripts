"""
NHL94 get-puck reward functions.
"""

from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_rf import (
    ClampCameraToPlayer,
    SampleRandomPos,
    SampleRandomPosAttackZone,
    SampleRandomPosDefenseZone,
    SampleRandomPosNeutralZone,
    _sample_position_away_from,
    _set_getpuck_puck_state,
)


def _init_getpuck_with_sampler(env, env_name, random_pos_fn):
    if env_name is None:
        env_name = 'NHL941on1-Genesis-v0'

    puck_holder_pos = random_pos_fn(env)

    if env_name == 'NHL941on1-Genesis-v0':
        x, y = _sample_position_away_from(env, random_pos_fn, puck_holder_pos, 18)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        env.set_value("p2_x", puck_holder_pos[0])
        env.set_value("p2_y", puck_holder_pos[1])
        positions = {"p1": (x, y), "p2": puck_holder_pos}
    elif env_name == 'NHL942on2-Genesis-v0':
        positions = {"p2": puck_holder_pos}
        x, y = _sample_position_away_from(env, random_pos_fn, puck_holder_pos, 18)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        positions["p1"] = (x, y)
        env.set_value("p2_x", puck_holder_pos[0])
        env.set_value("p2_y", puck_holder_pos[1])
        x, y = random_pos_fn(env)
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        positions["p2_2"] = (x, y)
        x, y = _sample_position_away_from(env, random_pos_fn, puck_holder_pos, 18)
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)
        positions["p1_2"] = (x, y)
    elif env_name == 'NHL94-Genesis-v0':
        positions = {"p2": puck_holder_pos}
        for i in range(5):
            x, y = _sample_position_away_from(env, random_pos_fn, puck_holder_pos, 18)
            if i == 0:
                env.set_value("p1_x", x)
                env.set_value("p1_y", y)
                positions["p1"] = (x, y)
            else:
                env.set_value(f"p1_{i+1}_x", x)
                env.set_value(f"p1_{i+1}_y", y)
                positions[f"p1_{i+1}"] = (x, y)

        env.set_value("p2_x", puck_holder_pos[0])
        env.set_value("p2_y", puck_holder_pos[1])
        for i in range(1, 5):
            x, y = random_pos_fn(env)
            env.set_value(f"p2_{i+1}_x", x)
            env.set_value(f"p2_{i+1}_y", y)
            positions[f"p2_{i+1}"] = (x, y)
    else:
        raise ValueError(f"Invalid environment name, got '{env_name}'")

    _set_getpuck_puck_state(env, env_name, puck_holder_pos)

    x, y = positions.get("p2", positions.get("p1", (0, 0)))
    camera_x, camera_y = ClampCameraToPlayer(x, y)
    env.set_value("camera_x", camera_x)
    env.set_value("camera_y", camera_y)


def init_getpuck(env, env_name=None):
    _init_getpuck_with_sampler(env, env_name, SampleRandomPos)


def init_getpuck_az(env, env_name=None):
    _init_getpuck_with_sampler(env, env_name, SampleRandomPosAttackZone)


def init_getpuck_nz(env, env_name=None):
    _init_getpuck_with_sampler(env, env_name, SampleRandomPosNeutralZone)


def init_getpuck_dz(env, env_name=None):
    _init_getpuck_with_sampler(env, env_name, SampleRandomPosDefenseZone)


def isdone_getpuck(state):
    t1 = state.team1
    t2 = state.team2
    if t1.player_haspuck is True or t1.goalie_haspuck is True:
        return True

    if t2.goalie_haspuck is True:
        return True

    if t2.stats.score > t2.last_stats.score:
        return True

    if state.time < 100:
        return True


def isdone_getpuck_az(state):
    if isdone_getpuck(state):
        return True

    if state.puck.y < 0:
        return True


def isdone_getpuck_nz(state):
    if isdone_getpuck(state):
        return True

    if state.puck.y < GameConsts.DEFENSEZONE_POS_Y:
        return True

    if state.puck.y >= GameConsts.ATACKZONE_POS_Y:
        return True


def isdone_getpuck_dz(state):
    if isdone_getpuck(state):
        return True

    if state.puck.y >= GameConsts.DEFENSEZONE_POS_Y:
        return True


def rf_getpuck(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0

    if t1.stats.bodychecks > t1.last_stats.bodychecks:
        rew = 0.5

    if t1.player_haspuck is True:
        return 1.0

    if t1.goalie_haspuck:
        rew = -1.0

    if t2.stats.score > t2.last_stats.score:
        rew = -1.0

    if t1.stats.shots > t1.last_stats.shots:
        rew = -1.0

    if state.puck.y <= 0:
        rew = -1.0

    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0
    dist_to_puck = t1.nz_players[controlled_idx].dist_to_puck

    rew -= 0.003
    rew += max(0.0, 1.0 - dist_to_puck) * 0.01

    return rew