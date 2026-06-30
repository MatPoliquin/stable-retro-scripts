"""
NHL94 Reward Functions
"""

import random
import numpy as np
from typing import Tuple, Callable
from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_mi import init_model, set_model_input

# =====================================================================
# Common functions
# =====================================================================
def RandomPos():
    return _sample_non_net_pos_uniform(-GameConsts.MAX_PLAYER_Y, GameConsts.MAX_PLAYER_Y)

def RandomPosAttackZone():
    return _sample_non_net_pos_uniform(GameConsts.ATACKZONE_POS_Y, GameConsts.MAX_PLAYER_Y)

def RandomPosDefenseZone():
    return _sample_non_net_pos_uniform(-GameConsts.MAX_PLAYER_Y, GameConsts.DEFENSEZONE_POS_Y - 1)

def _env_rng(env):
    return getattr(env, "np_random", None)

def _randint_inclusive(rng, low, high):
    if rng is None:
        return random.randint(low, high)

    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))

    return int(rng.randint(low, high + 1))

def _random_pos_from_rng(rng, y_min, y_max):
    half_width = GameConsts.SPAWNABLE_AREA_WIDTH // 2
    x = _randint_inclusive(rng, -half_width, half_width)
    y = _randint_inclusive(rng, y_min, y_max)
    return x, y

def _is_inside_net_area(x, y):
    in_goal_width = GameConsts.P1_NET_LEFT_POLL <= x <= GameConsts.P1_NET_RIGHT_POLL
    if not in_goal_width:
        return False

    # Treat positions on/behind each goal line as inside the net.
    in_p1_net = (GameConsts.P1_NET_Y - GameConsts.NET_DEPTH) <= y <= (GameConsts.P1_NET_Y + 1)
    in_p2_net = (GameConsts.P2_NET_Y - 1) <= y <= (GameConsts.P2_NET_Y + GameConsts.NET_DEPTH)
    return in_p1_net or in_p2_net

def _sample_non_net_pos_from_rng(env, y_min, y_max):
    rng = _env_rng(env)
    for _ in range(64):
        x, y = _random_pos_from_rng(rng, y_min, y_max)
        if not _is_inside_net_area(x, y):
            return x, y

    # Fallback: preserve progress even in pathological edge cases.
    return _random_pos_from_rng(rng, y_min, y_max)

def _sample_non_net_pos_uniform(y_min, y_max):
    half_width = GameConsts.SPAWNABLE_AREA_WIDTH * 0.5
    for _ in range(64):
        x = random.uniform(-half_width, half_width)
        y = random.uniform(y_min, y_max)
        if not _is_inside_net_area(x, y):
            return x, y

    return random.uniform(-half_width, half_width), random.uniform(y_min, y_max)

def _choose_index(env, count):
    rng = _env_rng(env)
    if count <= 1:
        return 0

    if rng is None:
        return random.randrange(count)

    if hasattr(rng, "integers"):
        return int(rng.integers(0, count))

    return int(rng.randint(0, count))

def SampleRandomPos(env):
    return _sample_non_net_pos_from_rng(env, -GameConsts.MAX_PLAYER_Y, GameConsts.MAX_PLAYER_Y)

def SampleRandomPosAttackZone(env):
    return _sample_non_net_pos_from_rng(env, GameConsts.ATACKZONE_POS_Y, GameConsts.MAX_PLAYER_Y)

def SampleRandomPosDefenseZone(env):
    return _sample_non_net_pos_from_rng(env, -GameConsts.MAX_PLAYER_Y, GameConsts.DEFENSEZONE_POS_Y - 1)

def SampleRandomPosNeutralZone(env):
    return _sample_non_net_pos_from_rng(
        env,
        GameConsts.DEFENSEZONE_POS_Y,
        GameConsts.ATACKZONE_POS_Y - 1,
    )

def ClampCameraToPlayer(pos_x, pos_y):
    camera_x = max(GameConsts.CAMERA_MIN_X, min(GameConsts.CAMERA_MAX_X, pos_x))
    camera_y = max(GameConsts.CAMERA_MIN_Y, min(GameConsts.CAMERA_MAX_Y, pos_y))
    return camera_x, camera_y

def input_overide(ac):
    if isinstance(ac, (list, np.ndarray)) and len(ac) == 12:
        ac[GameConsts.INPUT_B] = 0
        ac[GameConsts.INPUT_C] = 0
    elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
        ac[2] = 0

def input_overide_no_shoot(ac):
    if isinstance(ac, (list, np.ndarray)) and len(ac) == 12:
        ac[GameConsts.INPUT_C] = 0
    elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
        if ac[2] == 2:
            ac[2] = 0

def input_overide_empty(ac):
    return

def SetRandomSkaterPositions(env, env_name, random_pos_fn):
    positions = {}

    if env_name == 'NHL941on1-Genesis-v0':
        x, y = random_pos_fn(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        positions["p1"] = (x, y)
        x, y = random_pos_fn(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        positions["p2"] = (x, y)
        return positions

    if env_name == 'NHL942on2-Genesis-v0':
        x, y = random_pos_fn(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        positions["p1"] = (x, y)
        x, y = random_pos_fn(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        positions["p2"] = (x, y)
        x, y = random_pos_fn(env)
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        positions["p2_2"] = (x, y)
        x, y = random_pos_fn(env)
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)
        positions["p1_2"] = (x, y)
        return positions

    if env_name == 'NHL94-Genesis-v0':
        for i in range(5):
            x, y = random_pos_fn(env)
            if i == 0:
                env.set_value("p1_x", x)
                env.set_value("p1_y", y)
                positions["p1"] = (x, y)
            else:
                env.set_value(f"p1_{i+1}_x", x)
                env.set_value(f"p1_{i+1}_y", y)
                positions[f"p1_{i+1}"] = (x, y)

        for i in range(5):
            x, y = random_pos_fn(env)
            if i == 0:
                env.set_value("p2_x", x)
                env.set_value("p2_y", y)
                positions["p2"] = (x, y)
            else:
                env.set_value(f"p2_{i+1}_x", x)
                env.set_value(f"p2_{i+1}_y", y)
                positions[f"p2_{i+1}"] = (x, y)
        return positions

    raise ValueError(f"Invalid environment name, got '{env_name}'")

def _safe_set_value(env, name, value):
    try:
        env.set_value(name, value)
    except Exception:
        return

def _sample_position_away_from(env, random_pos_fn, avoid_pos, min_distance):
    if avoid_pos is None:
        return random_pos_fn(env)

    for _ in range(32):
        x, y = random_pos_fn(env)
        if GameConsts.Distance((x, y), avoid_pos) >= min_distance:
            return x, y

    return random_pos_fn(env)

def _set_getpuck_puck_state(env, env_name, puck_holder_pos):
    puck_x, puck_y = puck_holder_pos

    _safe_set_value(env, "puck_x", puck_x)
    _safe_set_value(env, "puck_y", puck_y)
    _safe_set_value(env, "puck_vel_x", 0)
    _safe_set_value(env, "puck_vel_y", 0)
    _safe_set_value(env, "fullstar_x", puck_x)
    _safe_set_value(env, "fullstar_y", puck_y)
    _safe_set_value(env, "p2_fullstar_x", puck_x)
    _safe_set_value(env, "p2_fullstar_y", puck_y)

    if env_name == 'NHL94-Genesis-v0':
        _safe_set_value(env, "puck_owner", 6)

def _set_team1_puck_state(env, env_name, puck_holder_pos, puck_owner_index):
    puck_x, puck_y = puck_holder_pos

    _safe_set_value(env, "puck_x", puck_x)
    _safe_set_value(env, "puck_y", puck_y)
    _safe_set_value(env, "puck_vel_x", 0)
    _safe_set_value(env, "puck_vel_y", 0)
    _safe_set_value(env, "fullstar_x", puck_x)
    _safe_set_value(env, "fullstar_y", puck_y)
    _safe_set_value(env, "emptystar_x", puck_x)
    _safe_set_value(env, "emptystar_y", puck_y)

    # Clear stale opponent possession/control markers from the source savestate.
    _safe_set_value(env, "p2_fullstar_x", 0)
    _safe_set_value(env, "p2_fullstar_y", -320)
    _safe_set_value(env, "p2_emptystar_x", 0)
    _safe_set_value(env, "p2_emptystar_y", -320)

    if env_name == 'NHL94-Genesis-v0':
        _safe_set_value(env, "puck_owner", puck_owner_index)

def init_attackzone(env, env_name):
    team1_positions = []

    if env_name == 'NHL941on1-Genesis-v0':
        x, y = SampleRandomPosAttackZone(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        team1_positions.append((x, y))
        x, y = SampleRandomPosAttackZone(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)

    elif env_name == 'NHL942on2-Genesis-v0':
        x, y = SampleRandomPosAttackZone(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        team1_positions.append((x, y))
        x, y = SampleRandomPosAttackZone(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        x, y = SampleRandomPosAttackZone(env)
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        x, y = SampleRandomPosAttackZone(env)
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)
        team1_positions.append((x, y))
    elif env_name == 'NHL94-Genesis-v0':
        # Team 1 players
        for i in range(5):
            x, y = SampleRandomPosAttackZone(env)
            #print(x,y)
            if i == 0:
                env.set_value("p1_x", x)
                env.set_value("p1_y", y)
            else:
                env.set_value(f"p1_{i+1}_x", x)
                env.set_value(f"p1_{i+1}_y", y)
            team1_positions.append((x, y))

        # Team 2 players
        for i in range(5):
            x, y = SampleRandomPosAttackZone(env)
            #x, y = RandomPosDefenseZone()
            if i == 0:
                env.set_value("p2_x", x)
                env.set_value("p2_y", y)
            else:
                env.set_value(f"p2_{i+1}_x", x)
                env.set_value(f"p2_{i+1}_y", y)
    else:
        raise ValueError(f"Invalid environment name, got '{env_name}'")

    puck_owner_index = _choose_index(env, len(team1_positions))
    _set_team1_puck_state(env, env_name, team1_positions[puck_owner_index], puck_owner_index)


def init_general(env, env_name):
    team1_positions = []

    if env_name == 'NHL941on1-Genesis-v0':
        x, y = SampleRandomPos(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        team1_positions.append((x, y))
        x, y = SampleRandomPos(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
    elif env_name == 'NHL942on2-Genesis-v0':
        x, y = SampleRandomPos(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        team1_positions.append((x, y))
        x, y = SampleRandomPos(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        x, y = SampleRandomPos(env)
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        x, y = SampleRandomPos(env)
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)
        team1_positions.append((x, y))
    elif env_name == 'NHL94-Genesis-v0':
        for i in range(5):
            x, y = SampleRandomPos(env)
            if i == 0:
                env.set_value("p1_x", x)
                env.set_value("p1_y", y)
            else:
                env.set_value(f"p1_{i+1}_x", x)
                env.set_value(f"p1_{i+1}_y", y)
            team1_positions.append((x, y))

        for i in range(5):
            x, y = SampleRandomPos(env)
            if i == 0:
                env.set_value("p2_x", x)
                env.set_value("p2_y", y)
            else:
                env.set_value(f"p2_{i+1}_x", x)
                env.set_value(f"p2_{i+1}_y", y)
    else:
        raise ValueError(f"Invalid environment name, got '{env_name}'")

    if team1_positions:
        puck_owner_index = _choose_index(env, len(team1_positions))
        _set_team1_puck_state(env, env_name, team1_positions[puck_owner_index], puck_owner_index)


from game_wrappers.nhl94.nhl94_rf_createopp import input_overide_createopp, isdone_createopp, rf_createopp
from game_wrappers.nhl94.nhl94_rf_defensezone import init_defensezone, isdone_defensezone, rf_defensezone
from game_wrappers.nhl94.nhl94_rf_general import isdone_general, init_general_v2, isdone_general_v2, rf_general, rf_general_v2
from game_wrappers.nhl94.nhl94_rf_getpuck import (
    init_getpuck,
    init_getpuck_az,
    init_getpuck_dz,
    init_getpuck_nz,
    isdone_getpuck,
    isdone_getpuck_az,
    isdone_getpuck_dz,
    isdone_getpuck_nz,
    rf_getpuck,
)
from game_wrappers.nhl94.nhl94_rf_keeppuck import init_keeppuck, isdone_keeppuck, rf_keeppuck
from game_wrappers.nhl94.nhl94_rf_passing import isdone_passing, rf_passing
from game_wrappers.nhl94.nhl94_rf_postplay import init_postplay, isdone_postplay, rf_postplay
from game_wrappers.nhl94.nhl94_rf_scoregoal import (
    isdone_crosscrease_v2,
    isdone_scoregoal,
    isdone_scoregoal_cc,
    isdone_scoregoal_ot,
    isdone_scoregoal_v4,
    isdone_scoregoal_v2,
    rf_crosscrease_v2,
    rf_scoregoal,
    rf_scoregoal_cc,
    rf_scoregoal_ot,
    rf_scoregoal_v4,
    rf_scoregoal_v2,
)
from game_wrappers.nhl94.nhl94_rf_selfplay import (
    init_selfplay,
    init_selfplay_defense,
    init_selfplay_offense,
    isdone_selfplay,
    isdone_selfplay_defense,
    isdone_selfplay_offense,
    rf_selfplay,
    rf_selfplay_defense,
    rf_selfplay_offense,
)

# =====================================================================
# Register Functions
# =====================================================================
MODEL_INPUT_FUNCTIONS = (init_model, set_model_input)

_reward_function_map = {
    "GetPuck": (init_getpuck, rf_getpuck, isdone_getpuck, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "GetPuckAZ": (init_getpuck_az, rf_getpuck, isdone_getpuck_az, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "GetPuckNZ": (init_getpuck_nz, rf_getpuck, isdone_getpuck_nz, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "GetPuckDZ": (init_getpuck_dz, rf_getpuck, isdone_getpuck_dz, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoalCC": (init_attackzone, rf_scoregoal_cc, isdone_scoregoal_cc, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "CrossCreaseV2": (init_attackzone, rf_crosscrease_v2, isdone_crosscrease_v2, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoalOT": (init_attackzone, rf_scoregoal_ot, isdone_scoregoal_ot, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoal": (init_attackzone, rf_scoregoal, isdone_scoregoal, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoalV2": (init_attackzone, rf_scoregoal_v2, isdone_scoregoal_v2, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoalV4": (init_attackzone, rf_scoregoal_v4, isdone_scoregoal_v4, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "CreateOpp": (init_attackzone, rf_createopp, isdone_createopp, *MODEL_INPUT_FUNCTIONS, input_overide_createopp),
    "KeepPuck": (init_keeppuck, rf_keeppuck, isdone_keeppuck, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "DefenseZone": (init_defensezone, rf_defensezone, isdone_defensezone, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "Passing": (init_attackzone, rf_passing, isdone_passing, *MODEL_INPUT_FUNCTIONS, input_overide_no_shoot),
    "General": (init_general, rf_general, isdone_general, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "GeneralV2": (init_general_v2, rf_general_v2, isdone_general_v2, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "PostPlay": (init_postplay, rf_postplay, isdone_postplay, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "SelfPlay": (init_selfplay, rf_selfplay, isdone_selfplay, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "SelfPlayOffenseFinetune": (init_selfplay_offense, rf_selfplay_offense, isdone_selfplay_offense, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "SelfPlayDefenseFinetune": (init_selfplay_defense, rf_selfplay_defense, isdone_selfplay_defense, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
}

def register_functions(name: str) -> Tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    if name not in _reward_function_map:
        raise ValueError(f"Unsupported Reward Function: {name}")
    return _reward_function_map[name]
