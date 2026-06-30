"""
NHL94 breakaway reward functions.
"""

import random

from game_wrappers.nhl94.nhl94_const import GameConsts


BREAKAWAY_GOAL_REWARD = 1.0
BREAKAWAY_TURNOVER_PENALTY = -0.5
BREAKAWAY_AGAINST_PENALTY = -1.0

_CARRIER_X_LANES = (
    (-95, -58),
    (-57, -24),
    (-23, 23),
    (24, 57),
    (58, 95),
)
_CARRIER_NEUTRAL_START_PERCENT = 35
_CARRIER_NEUTRAL_Y_MIN = GameConsts.DEFENSEZONE_POS_Y + 20
_CARRIER_NEUTRAL_Y_MAX = GameConsts.ATACKZONE_POS_Y - 1
_CARRIER_ATTACK_Y_MIN = GameConsts.ATACKZONE_POS_Y
_CARRIER_ATTACK_Y_MAX = GameConsts.ATACKZONE_POS_Y + 35
_CARRIER_FORWARD_VEL_MIN = 8
_CARRIER_FORWARD_VEL_MAX = 18
_CARRIER_CENTERING_VEL_MIN = 2
_CARRIER_CENTERING_VEL_MAX = 7
_TRAILING_Y_MIN = GameConsts.DEFENSEZONE_POS_Y
_TRAILING_CARRIER_Y_GAP = 24
_MIN_TRAILING_DISTANCE = 24
_GOALIE_X_MIN = -8
_GOALIE_X_MAX = 8
_GOALIE_Y_MIN = GameConsts.CREASE_LOWER_BOUND
_GOALIE_Y_MAX = GameConsts.CREASE_UPPER_BOUND


def _env_rng(env):
    return getattr(env, "np_random", None)


def _randint_inclusive(rng, low, high):
    if rng is None:
        return random.randint(low, high)

    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))

    return int(rng.randint(low, high + 1))


def _safe_set_value(env, name, value):
    try:
        env.set_value(name, value)
    except Exception:
        return


def _clamp_camera_to_player(pos_x, pos_y):
    camera_x = max(GameConsts.CAMERA_MIN_X, min(GameConsts.CAMERA_MAX_X, pos_x))
    camera_y = max(GameConsts.CAMERA_MIN_Y, min(GameConsts.CAMERA_MAX_Y, pos_y))
    return camera_x, camera_y


def _set_team1_puck_state(
    env,
    env_name,
    puck_holder_pos,
    puck_owner_index,
    puck_vel,
):
    puck_x, puck_y = puck_holder_pos
    puck_vel_x, puck_vel_y = puck_vel

    _safe_set_value(env, "puck_x", puck_x)
    _safe_set_value(env, "puck_y", puck_y)
    _safe_set_value(env, "puck_vel_x", puck_vel_x)
    _safe_set_value(env, "puck_vel_y", puck_vel_y)
    _safe_set_value(env, "fullstar_x", puck_x)
    _safe_set_value(env, "fullstar_y", puck_y)
    _safe_set_value(env, "emptystar_x", puck_x)
    _safe_set_value(env, "emptystar_y", puck_y)
    _safe_set_value(env, "p1_emptystar_x", puck_x)
    _safe_set_value(env, "p1_emptystar_y", puck_y)

    _safe_set_value(env, "p2_fullstar_x", 0)
    _safe_set_value(env, "p2_fullstar_y", -320)
    _safe_set_value(env, "p2_emptystar_x", 0)
    _safe_set_value(env, "p2_emptystar_y", -320)

    _safe_set_value(env, "puck_owner", puck_owner_index)


def _player_names(env_name):
    if env_name == 'NHL941on1-Genesis-v0':
        return ['p1'], ['p2']

    if env_name == 'NHL942on2-Genesis-v0':
        return ['p1', 'p1_2'], ['p2', 'p2_2']

    if env_name == 'NHL94-Genesis-v0':
        return (
            ['p1', 'p1_2', 'p1_3', 'p1_4', 'p1_5'],
            ['p2', 'p2_2', 'p2_3', 'p2_4', 'p2_5'],
        )

    raise ValueError(f"Invalid environment name, got '{env_name}'")


def _set_skater_position(env, player_name, pos, vel=(0, 0)):
    x, y = pos
    vel_x, vel_y = vel
    env.set_value(f"{player_name}_x", x)
    env.set_value(f"{player_name}_y", y)
    _safe_set_value(env, f"{player_name}_vel_x", vel_x)
    _safe_set_value(env, f"{player_name}_vel_y", vel_y)


def _sample_breakaway_carrier_pos(env):
    rng = _env_rng(env)
    lane_index = _randint_inclusive(rng, 0, len(_CARRIER_X_LANES) - 1)
    x_min, x_max = _CARRIER_X_LANES[lane_index]
    neutral_start = _randint_inclusive(rng, 1, 100) <= _CARRIER_NEUTRAL_START_PERCENT
    y_min = _CARRIER_NEUTRAL_Y_MIN if neutral_start else _CARRIER_ATTACK_Y_MIN
    y_max = _CARRIER_NEUTRAL_Y_MAX if neutral_start else _CARRIER_ATTACK_Y_MAX

    return (
        _randint_inclusive(rng, x_min, x_max),
        _randint_inclusive(rng, y_min, y_max),
    )


def _sample_breakaway_carrier_vel(env, carrier_pos):
    rng = _env_rng(env)
    carrier_x, _ = carrier_pos
    vel_y = _randint_inclusive(
        rng,
        _CARRIER_FORWARD_VEL_MIN,
        _CARRIER_FORWARD_VEL_MAX,
    )

    if carrier_x < -45:
        vel_x = _randint_inclusive(
            rng,
            _CARRIER_CENTERING_VEL_MIN,
            _CARRIER_CENTERING_VEL_MAX,
        )
    elif carrier_x > 45:
        vel_x = -_randint_inclusive(
            rng,
            _CARRIER_CENTERING_VEL_MIN,
            _CARRIER_CENTERING_VEL_MAX,
        )
    else:
        vel_x = _randint_inclusive(
            rng,
            -_CARRIER_CENTERING_VEL_MIN,
            _CARRIER_CENTERING_VEL_MIN,
        )

    return vel_x, vel_y


def _sample_trailing_pos(env, carrier_pos):
    rng = _env_rng(env)
    half_width = GameConsts.SPAWNABLE_AREA_WIDTH // 2
    _, carrier_y = carrier_pos
    trailing_y_max = min(
        GameConsts.ATACKZONE_POS_Y - 1,
        carrier_y - _TRAILING_CARRIER_Y_GAP,
    )
    trailing_y_min = min(_TRAILING_Y_MIN, trailing_y_max)

    for _ in range(64):
        x = _randint_inclusive(rng, -half_width, half_width)
        y = _randint_inclusive(rng, trailing_y_min, trailing_y_max)
        if GameConsts.Distance((x, y), carrier_pos) >= _MIN_TRAILING_DISTANCE:
            return x, y

    return (
        _randint_inclusive(rng, -half_width, half_width),
        _randint_inclusive(rng, trailing_y_min, trailing_y_max),
    )


def _set_opponent_goalie(env):
    rng = _env_rng(env)
    goalie_x = _randint_inclusive(rng, _GOALIE_X_MIN, _GOALIE_X_MAX)
    goalie_y = _randint_inclusive(rng, _GOALIE_Y_MIN, _GOALIE_Y_MAX)
    _safe_set_value(env, "g2_x", goalie_x)
    _safe_set_value(env, "g2_y", goalie_y)
    _safe_set_value(env, "g2_vel_x", 0)
    _safe_set_value(env, "g2_vel_y", 0)


def init_breakaway(env, env_name=None):
    if env_name is None:
        env_name = 'NHL941on1-Genesis-v0'

    team1_names, team2_names = _player_names(env_name)
    carrier_index = 0
    carrier_pos = _sample_breakaway_carrier_pos(env)
    carrier_vel = _sample_breakaway_carrier_vel(env, carrier_pos)

    for index, player_name in enumerate(team1_names):
        pos = carrier_pos if index == carrier_index else _sample_trailing_pos(env, carrier_pos)
        vel = carrier_vel if index == carrier_index else (0, 0)
        _set_skater_position(env, player_name, pos, vel)

    for player_name in team2_names:
        _set_skater_position(env, player_name, _sample_trailing_pos(env, carrier_pos))

    _set_opponent_goalie(env)
    _set_team1_puck_state(env, env_name, carrier_pos, carrier_index, carrier_vel)

    camera_x, camera_y = _clamp_camera_to_player(*carrier_pos)
    _safe_set_value(env, "camera_x", camera_x)
    _safe_set_value(env, "camera_y", camera_y)


def _breakaway_active(state):
    t1 = state.team1
    carrier = t1.get_possession_player()
    if carrier is None or carrier is t1.goalie:
        carrier = t1.get_controlled_player()

    return bool(carrier.is_breakaway or state.engine.breakaway_context)


def _breakaway_seen(state):
    if _breakaway_active(state):
        state.breakaway_seen = True

    return bool(getattr(state, "breakaway_seen", False))


def isdone_breakaway(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score:
        return True

    if t2.stats.score > t2.last_stats.score:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if t1.goalie_haspuck:
        return True

    if state.puck.y >= GameConsts.ATACKZONE_POS_Y:
        state.breakaway_entered_attack_zone = True

    if (
        getattr(state, "breakaway_entered_attack_zone", False)
        and state.puck.y < GameConsts.ATACKZONE_POS_Y
    ):
        return True

    if state.puck.y < GameConsts.DEFENSEZONE_POS_Y:
        return True

    if state.time < 100:
        return True

    return False


def rf_breakaway(state):
    t1 = state.team1
    t2 = state.team2
    breakaway_seen = _breakaway_seen(state)

    if t1.stats.score > t1.last_stats.score:
        return BREAKAWAY_GOAL_REWARD if breakaway_seen else 0.0

    if t2.stats.score > t2.last_stats.score:
        return BREAKAWAY_AGAINST_PENALTY

    if t2.player_haspuck or t2.goalie_haspuck:
        return BREAKAWAY_TURNOVER_PENALTY

    return 0.0