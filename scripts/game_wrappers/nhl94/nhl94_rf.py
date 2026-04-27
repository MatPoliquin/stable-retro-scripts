"""
NHL94 Reward Functions
"""

import random
import numpy as np
import math
from typing import Tuple, Callable
from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_mi import init_model, init_model_rel, init_model_rel_dist, init_model_rel_dist_buttons, init_model_1p, init_model_2p, \
      set_model_input, set_model_input_1p, set_model_input_2p, set_model_input_rel, set_model_input_rel_dist, set_model_input_rel_dist_buttons, \
    init_model_invariant, set_model_input_invariant, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2

# =====================================================================
# Common functions
# =====================================================================
def RandomPos():
    x = (random.random() - 0.5) * GameConsts.SPAWNABLE_AREA_WIDTH
    y = (random.random() - 0.5) * GameConsts.SPAWNABLE_AREA_HEIGHT

    #print(x,y)
    return x, y

def RandomPosAttackZone():
    x = (random.random() - 0.5) * GameConsts.SPAWNABLE_AREA_WIDTH
    y = (random.random() * (GameConsts.SPAWNABLE_ZONE_HEIGHT + 100)) + 0

    #print(x,y)
    return x, y

def RandomPosDefenseZone():
    x = (random.random() - 0.5) * GameConsts.SPAWNABLE_AREA_WIDTH
    y = (random.random() * GameConsts.SPAWNABLE_ZONE_HEIGHT) - 220

    #print(x,y)
    return x, y

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
    half_height = GameConsts.SPAWNABLE_AREA_HEIGHT // 2
    return _random_pos_from_rng(_env_rng(env), -half_height, half_height)

def SampleRandomPosAttackZone(env):
    return _random_pos_from_rng(_env_rng(env), 0, GameConsts.SPAWNABLE_ZONE_HEIGHT + 99)

def SampleRandomPosDefenseZone(env):
    return _random_pos_from_rng(_env_rng(env), -220, -81)

def SampleRandomPosNeutralZone(env):
    return _random_pos_from_rng(
        _env_rng(env),
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


# =====================================================================
# General - One model for both offense and defense
# =====================================================================
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

    if team1_positions:
        puck_owner_index = _choose_index(env, len(team1_positions))
        _set_team1_puck_state(env, env_name, team1_positions[puck_owner_index], puck_owner_index)

def isdone_general(state):
    t1 = state.team1
    t2 = state.team2

    if state.time < 100:
        return True

    if t1.stats.score > t1.last_stats.score:
        return True
    if t2.stats.score > t2.last_stats.score:
        return True

    return False

def rf_general(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0
    if state.puck.y > 100 and t1.player_haspuck:
        rew = 0.1
    
    if t1.stats.passing > t1.last_stats.passing:
        rew += 0.1

    if state.puck.y > 120 and t1.stats.onetimer > t1.last_stats.onetimer:
        rew += 0.5

    if t1.stats.score > t1.last_stats.score:
        rew = 1.0

    # Scoring rewards (team 1 scores or concedes)
    if t1.stats.score > t1.last_stats.score:
        return 1.0  # Big reward for scoring
    if t2.stats.score > t2.last_stats.score:
        return -1.0  # Penalty for conceding

    if t2.stats.shots > t2.last_stats.shots:
        return -0.5  # Penalty for conceding

    return rew

# =====================================================================
# ScoreGoal - Cross Crease technic
# =====================================================================
def isdone_scoregoal_cc(state):
    t1 = state.team1
    t2 = state.team2

    # Success
    if t1.stats.score > t1.last_stats.score:
        return True

    # Mild failures (just end episode)
    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.puck.y < 100:
        return True

    # Timeout
    if state.time < 100:
        return True

    return False

def rf_scoregoal_cc(state):
    t1 = state.team1
    t2 = state.team2
    rew = 0.0



    # Failure conditions
    if t2.player_haspuck or t2.goalie_haspuck:
        return -0.5  # Less punitive

    if state.puck.y < 100:
        return -0.5

    # Get controlled player
    controlled_player = t1.players[t1.control-1] if t1.control > 0 else t1.goalie

    # Check crease position
    in_crease = (
        controlled_player.y < GameConsts.CREASE_UPPER_BOUND and
        controlled_player.y > GameConsts.CREASE_LOWER_BOUND
    )

    # Reward for entering crease area
    if in_crease:
        #rew += 0.1

        if abs(controlled_player.x) < GameConsts.CREASE_MAX_X * 3:
            rew += 0.1

        # Extra reward for proper positioning (lateral movement)
        if abs(controlled_player.x) < GameConsts.CREASE_MAX_X:
            rew += 0.4

        # Reward for velocity toward net
        #if controlled_player.vy > 0:  # Moving up toward opponent net
        #    rew += 0.1 * min(controlled_player.vy, 1.0)  # Cap velocity reward

        if abs(controlled_player.vx) > 15:  # Moving up toward opponent net
            rew = 0.5

    # Reward for shooting when in good position
    if state.action[5] == 1 and in_crease:  # C button pressed (shoot)
        if abs(controlled_player.x - t2.goalie.x) > GameConsts.CREASE_MIN_GOALIE_PUCK_DIST_X:
            rew += 1.0  # Big reward for shooting from good position
        else:
            rew += 0.2  # Small reward for attempting shot

    # Reward for passing (helps set up cross-crease plays)
    if t1.stats.passing > t1.last_stats.passing:
        rew += 0.5

    # Success condition
    if t1.stats.score > 0:
        return 1.0  # Big reward for scoring

    return rew

# =====================================================================
# ScoreGoal - One Timers
# =====================================================================
def isdone_scoregoal_ot(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.puck.y < 0:
        return True

    if state.time < 100:
        return True

    return False

def rf_scoregoal_ot(state):
    t1 = state.team1
    t2 = state.team2
    rew = 0.0

    if t1.stats.passing > t1.last_stats.passing:
        rew = 0.01

    #if t1.stats.shots > t1.last_stats.shots:
    #    rew = 0.1

    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew = 0.1

    if t1.stats.score > t1.last_stats.score:
        rew = 1.0



    return rew

# =====================================================================
# ScoreGoal
# =====================================================================
def isdone_scoregoal(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.puck.y < 100:
        return True

    if state.time < 100:
        return True

    return False

def rf_scoregoal(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    #if t2.player_haspuck or t2.goalie_haspuck:
    #    rew = -0.5

    #if state.puck.y < 100:
    #    rew = -1.0

    if t1.stats.passing > t1.last_stats.passing:
        rew = 0.1

    #if t1.stats.shots > t1.last_stats.shots:
    #    rew = 0.1

    #if t1.stats.score > t1.last_stats.score:
    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew = 0.1

    if t1.stats.score > t1.last_stats.score:
        rew = 1.0

    return rew


def isdone_scoregoal_v2(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score:
        return True

    if state.puck.y < GameConsts.ATACKZONE_POS_Y:
        return True

    if state.time < 100:
        return True
    
    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    tracker = getattr(state, "_scoregoal_v2_tracker", None)
    if tracker is None:
        return False

    if tracker["stall_frames"] >= 45:
        return True

    if tracker["possession_frames"] >= 240 and not tracker["shot_taken_once"]:
        return True

    return False


def rf_scoregoal_v2(state):
    t1 = state.team1
    t2 = state.team2
    engine = state.engine

    tracker = getattr(state, "_scoregoal_v2_tracker", None)
    if tracker is None:
        tracker = {
            "possession_frames": 0,
            "stall_frames": 0,
            "shot_mode_frames": 0,
            "best_attack_y": state.puck.y,
            "last_puck_x": state.puck.x,
            "last_puck_y": state.puck.y,
            "prev_shot_mode_active": bool(engine.shot_mode_active),
            "prev_shot_taken": bool(engine.shot_taken),
            "prev_top_shelf": bool(engine.in_close_top_shelf),
            "prev_in_danger_area": False,
            "prev_team1_haspuck": bool(t1.player_haspuck or t1.goalie_haspuck),
            "prev_team2_haspuck": bool(t2.player_haspuck or t2.goalie_haspuck),
            "shot_taken_once": False,
        }
        setattr(state, "_scoregoal_v2_tracker", tracker)

    if t1.stats.score > t1.last_stats.score:
        return 1.0

    if t2.player_haspuck or t2.goalie_haspuck:
        tracker["possession_frames"] = 0
        tracker["shot_mode_frames"] = 0
        tracker["stall_frames"] = 0
        tracker["best_attack_y"] = max(GameConsts.ATACKZONE_POS_Y, state.puck.y)
        tracker["prev_shot_mode_active"] = bool(engine.shot_mode_active)
        tracker["prev_shot_taken"] = bool(engine.shot_taken)
        tracker["prev_top_shelf"] = bool(engine.in_close_top_shelf)
        tracker["prev_in_danger_area"] = False
        tracker["last_puck_x"] = state.puck.x
        tracker["last_puck_y"] = state.puck.y
        tracker["prev_team1_haspuck"] = False
        tracker["prev_team2_haspuck"] = True
        return -0.05

    if state.puck.y < GameConsts.ATACKZONE_POS_Y:
        exited_after_team1_control = tracker["prev_team1_haspuck"] and not tracker["prev_team2_haspuck"]
        return -1.0 if exited_after_team1_control else -0.5

    controlled_player = t1.get_controlled_player()
    opponent_goalie = t2.goalie
    controlled_idx = t1.control - 1 if t1.control > 0 else -1

    in_slot_lane = abs(controlled_player.x) < GameConsts.CREASE_MAX_X * 3
    near_net = controlled_player.y > (GameConsts.CREASE_LOWER_BOUND - 10)
    in_danger_area = near_net and abs(controlled_player.x) < GameConsts.CREASE_MAX_X * 2
    goalie_gap_x = abs(controlled_player.x - opponent_goalie.x)
    goalie_gap_good = goalie_gap_x > GameConsts.CREASE_MIN_GOALIE_PUCK_DIST_X
    goalie_committed = opponent_goalie.is_pad_stack or opponent_goalie.is_dive
    control_speed = math.hypot(controlled_player.vx, controlled_player.vy)
    puck_step = math.hypot(state.puck.x - tracker["last_puck_x"], state.puck.y - tracker["last_puck_y"])
    corner_trap = controlled_player.y > 150 and abs(controlled_player.x) > 55
    shot_taken_now = bool(engine.shot_taken) and not tracker["prev_shot_taken"]
    shot_mode_started = bool(engine.shot_mode_active) and not tracker["prev_shot_mode_active"]
    top_shelf_started = bool(engine.in_close_top_shelf) and not tracker["prev_top_shelf"]

    teammate_one_timer = False
    teammate_one_timer_lane = False
    for index, player in enumerate(t1.players):
        if index == controlled_idx:
            continue
        teammate_one_timer = teammate_one_timer or bool(player.is_one_timer)
        teammate_one_timer_lane = teammate_one_timer_lane or bool(player.is_one_timer and player.passing_lane_clear)

    rew = 0.0

    if t1.player_haspuck:
        tracker["possession_frames"] += 1

        attack_progress = max(0.0, state.puck.y - tracker["best_attack_y"])
        if attack_progress > 0:
            rew += min(attack_progress * 0.01, 0.10)
            tracker["best_attack_y"] = state.puck.y

        rew -= 0.003

        if tracker["possession_frames"] > 60:
            rew -= min((tracker["possession_frames"] - 60) * 0.001, 0.06)

        if corner_trap:
            rew -= 0.03

        if puck_step < 2.0 and control_speed < 4.0 and not engine.shot_mode_active and not state.action[4] and not state.action[5]:
            tracker["stall_frames"] += 1
        else:
            tracker["stall_frames"] = max(0, tracker["stall_frames"] - 2)

        if tracker["stall_frames"] > 15:
            rew -= min((tracker["stall_frames"] - 15) * 0.004, 0.12)
    else:
        tracker["possession_frames"] = 0
        tracker["shot_mode_frames"] = 0
        tracker["stall_frames"] = 0
        tracker["best_attack_y"] = max(GameConsts.ATACKZONE_POS_Y, state.puck.y)

    if in_danger_area and not tracker["prev_in_danger_area"]:
        rew += 0.08

    if t1.stats.passing > t1.last_stats.passing:
        rew += 0.15
        if teammate_one_timer_lane:
            rew += 0.10
        tracker["stall_frames"] = max(0, tracker["stall_frames"] - 10)

    if teammate_one_timer and teammate_one_timer_lane and not t1.player_haspuck:
        rew += 0.03

    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew += 0.35

    if engine.shot_mode_active:
        tracker["shot_mode_frames"] += 1
        if shot_mode_started and t1.player_haspuck and in_slot_lane:
            rew += 0.10
            if goalie_gap_good:
                rew += 0.05
            if goalie_committed:
                rew += 0.05

        if tracker["shot_mode_frames"] > 12:
            rew -= min((tracker["shot_mode_frames"] - 12) * 0.01, 0.12)
    else:
        tracker["shot_mode_frames"] = 0

    if shot_taken_now:
        tracker["shot_taken_once"] = True
        rew += 0.10
        if in_danger_area:
            rew += 0.08
        if goalie_gap_good:
            rew += 0.08
        if engine.goalie_box_small:
            rew += 0.12
        if goalie_committed:
            rew += 0.10
        if engine.controlled_is_shooter:
            rew += min(engine.pass_speed / 512.0, 0.08)
        tracker["stall_frames"] = 0

    if top_shelf_started and near_net:
        rew += 0.18

    if shot_taken_now and engine.one_timer_collision_mode:
        rew += 0.10

    if opponent_goalie.is_pad_stack and near_net and shot_taken_now:
        rew += 0.10

    if state.action[5] and not in_slot_lane:
        rew -= 0.02

    if shot_taken_now and not goalie_committed and not goalie_gap_good:
        rew -= 0.05

    tracker["prev_shot_mode_active"] = bool(engine.shot_mode_active)
    tracker["prev_shot_taken"] = bool(engine.shot_taken)
    tracker["prev_top_shelf"] = bool(engine.in_close_top_shelf)
    tracker["prev_in_danger_area"] = in_danger_area
    tracker["last_puck_x"] = state.puck.x
    tracker["last_puck_y"] = state.puck.y
    tracker["prev_team1_haspuck"] = bool(t1.player_haspuck or t1.goalie_haspuck)
    tracker["prev_team2_haspuck"] = bool(t2.player_haspuck or t2.goalie_haspuck)

    return rew


def _x_side(value: float, threshold: float = 6.0) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def isdone_crosscrease_v2(state):
    if isdone_scoregoal(state):
        return True

    tracker = getattr(state, "_crosscrease_v2_tracker", None)
    if tracker is None:
        return False

    if tracker["stall_frames"] >= 35:
        return True

    if tracker["possession_frames"] >= 220 and not tracker["shot_taken_once"]:
        return True

    if tracker["cross_pass_window"] == 0 and tracker["cross_pass_completed"] and not tracker["shot_taken_once"]:
        return True

    return False


def rf_crosscrease_v2(state):
    t1 = state.team1
    t2 = state.team2
    engine = state.engine

    tracker = getattr(state, "_crosscrease_v2_tracker", None)
    if tracker is None:
        tracker = {
            "possession_frames": 0,
            "stall_frames": 0,
            "best_attack_y": state.puck.y,
            "last_puck_x": state.puck.x,
            "last_puck_y": state.puck.y,
            "prev_shot_taken": bool(engine.shot_taken),
            "prev_top_shelf": bool(engine.in_close_top_shelf),
            "setup_side": 0,
            "max_goalie_pull": 0.0,
            "commit_window": 0,
            "cross_pass_window": 0,
            "cross_pass_completed": False,
            "shot_taken_once": False,
        }
        setattr(state, "_crosscrease_v2_tracker", tracker)

    if t1.stats.score > t1.last_stats.score:
        return 1.0

    if t2.player_haspuck or t2.goalie_haspuck:
        return -0.6

    if state.puck.y < GameConsts.ATACKZONE_POS_Y:
        return -0.5

    controlled_player = t1.get_controlled_player()
    opponent_goalie = t2.goalie
    control_speed = math.hypot(controlled_player.vx, controlled_player.vy)
    puck_step = math.hypot(state.puck.x - tracker["last_puck_x"], state.puck.y - tracker["last_puck_y"])
    control_side = _x_side(controlled_player.x)
    puck_side = _x_side(state.puck.x)
    goalie_committed = opponent_goalie.is_pad_stack or opponent_goalie.is_dive
    shot_taken_now = bool(engine.shot_taken) and not tracker["prev_shot_taken"]
    top_shelf_started = bool(engine.in_close_top_shelf) and not tracker["prev_top_shelf"]

    near_net = controlled_player.y > (GameConsts.CREASE_LOWER_BOUND - 10)
    side_setup = near_net and abs(controlled_player.x) > GameConsts.CREASE_MAX_X and abs(controlled_player.x) < 75
    deep_side_setup = side_setup and abs(controlled_player.x) > (GameConsts.CREASE_MAX_X + 10)
    danger_receiver_area = controlled_player.y > GameConsts.CREASE_LOWER_BOUND and abs(controlled_player.x) < GameConsts.CREASE_MAX_X * 2
    corner_trap = controlled_player.y > 150 and abs(controlled_player.x) > 65

    rew = 0.0

    if t1.player_haspuck:
        tracker["possession_frames"] += 1
        tracker["best_attack_y"] = max(tracker["best_attack_y"], state.puck.y)

        rew -= 0.004

        if tracker["possession_frames"] > 70:
            rew -= min((tracker["possession_frames"] - 70) * 0.0015, 0.08)

        if corner_trap:
            rew -= 0.04

        if puck_step < 2.0 and control_speed < 4.0 and not state.action[4] and not state.action[5]:
            tracker["stall_frames"] += 1
        else:
            tracker["stall_frames"] = max(0, tracker["stall_frames"] - 2)

        if tracker["stall_frames"] > 12:
            rew -= min((tracker["stall_frames"] - 12) * 0.005, 0.14)
    else:
        tracker["possession_frames"] = 0
        tracker["stall_frames"] = 0

    if side_setup and control_side != 0:
        if tracker["setup_side"] != control_side:
            tracker["setup_side"] = control_side
            tracker["max_goalie_pull"] = abs(opponent_goalie.x)
            rew += 0.08

        if deep_side_setup:
            rew += 0.02

        goalie_pull = abs(opponent_goalie.x)
        if goalie_pull > tracker["max_goalie_pull"] + 3:
            rew += min((goalie_pull - tracker["max_goalie_pull"]) * 0.01, 0.08)
            tracker["max_goalie_pull"] = goalie_pull

    if goalie_committed and side_setup:
        if tracker["commit_window"] == 0:
            rew += 0.20
        tracker["commit_window"] = 30

    if top_shelf_started and near_net:
        rew += 0.18
        tracker["commit_window"] = max(tracker["commit_window"], 30)

    if t1.stats.passing > t1.last_stats.passing:
        rew += 0.06
        if tracker["setup_side"] != 0 and puck_side != 0 and puck_side != tracker["setup_side"] and danger_receiver_area:
            rew += 0.30
            if tracker["commit_window"] > 0:
                rew += 0.15
            tracker["cross_pass_window"] = 30
            tracker["cross_pass_completed"] = True
            tracker["stall_frames"] = 0
        else:
            tracker["cross_pass_window"] = 0

    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew += 0.12
        if tracker["cross_pass_window"] > 0:
            rew += 0.25

    if shot_taken_now:
        tracker["shot_taken_once"] = True
        rew += 0.05

        if tracker["cross_pass_window"] > 0:
            rew += 0.35
        elif tracker["commit_window"] > 0 and near_net:
            rew += 0.15

        if engine.one_timer_collision_mode:
            rew += 0.10
        if engine.in_close_top_shelf:
            rew += 0.12
        if goalie_committed:
            rew += 0.10

        tracker["cross_pass_window"] = 0
        tracker["commit_window"] = 0
        tracker["stall_frames"] = 0

    if tracker["commit_window"] > 0:
        tracker["commit_window"] -= 1

    if tracker["cross_pass_window"] > 0:
        tracker["cross_pass_window"] -= 1

    if state.action[5] and not danger_receiver_area and tracker["cross_pass_window"] == 0:
        rew -= 0.03

    tracker["prev_shot_taken"] = bool(engine.shot_taken)
    tracker["prev_top_shelf"] = bool(engine.in_close_top_shelf)
    tracker["last_puck_x"] = state.puck.x
    tracker["last_puck_y"] = state.puck.y

    return rew

# =====================================================================
# KeepPuck
# =====================================================================
def init_keeppuck(env, env_name=None):
    #x, y = self.RandomPos()
    #self.env.set_value("rpuck_x", x)
    #self.env.set_value("rpuck_y", y)
    if env_name is None:
        env_name = 'NHL941on1-Genesis-v0'

    SetRandomSkaterPositions(env, env_name, SampleRandomPos)

def isdone_keeppuck(state):
    t1 = state.team1
    t2 = state.team2

    if t1.player_haspuck == False:
        return True

    if state.puck.y < 100:
        return True

def rf_keeppuck(state):
    t1 = state.team1
    t2 = state.team2

    rew = 1.0
    if not t1.player_haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    # Encourage movement; penalize stagnation
    # Use controlled skater if available, otherwise first skater
    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0
    player = t1.players[controlled_idx]

    speed = math.sqrt(player.vx**2 + player.vy**2)
    speed_norm = min(1.0, speed / GameConsts.MAX_VEL_XY)

    motion_bonus = 0.5 * speed_norm         # scale movement reward
    low_speed_threshold = 0.15               # ~7.5% of max speed
    stagnation_penalty = 0.5 if speed_norm < low_speed_threshold else 0.0

    if t1.player_haspuck and state.puck.y >= 100:
        #rew += motion_bonus
        rew -= stagnation_penalty

    if state.puck.x >= 125 or state.puck.x <= -125:
        rew = -0.5


    return rew

# =====================================================================
# GetPuck
# =====================================================================
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
    if t1.player_haspuck == True or t1.goalie_haspuck == True:
        return True

    if t2.goalie_haspuck == True:
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

    #if t2.player_haspuck == True:
    #    rew -= 0.1

    if t1.player_haspuck == True:
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
    dist_to_puck = t1.nz_players[controlled_idx].dist_to_puck  # 0.0 to ~1.0

    # Small time pressure: ~0.3 over 100 frames
    rew -= 0.003

    # Closeness bonus: reward being near the puck (always positive or zero)
    rew += max(0.0, 1.0 - dist_to_puck) * 0.01

    return rew

# =====================================================================
# DefenseZone
# =====================================================================
def init_defensezone(env, env_name):
    if env_name == 'NHL941on1-Genesis-v0':
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)

    elif env_name == 'NHL942on2-Genesis-v0':
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)
    elif env_name == 'NHL94-Genesis-v0':
        # Team 1 players
        for i in range(5):
            x, y = SampleRandomPosDefenseZone(env)
            #print(x,y)
            if i == 0:
                env.set_value("p1_x", x)
                env.set_value("p1_y", y)
            else:
                env.set_value(f"p1_{i+1}_x", x)
                env.set_value(f"p1_{i+1}_y", y)

        # Team 2 players
        for i in range(5):
            x, y = SampleRandomPosDefenseZone(env)
            #x, y = RandomPosDefenseZone()
            if i == 0:
                env.set_value("p2_x", x)
                env.set_value("p2_y", y)
            else:
                env.set_value(f"p2_{i+1}_x", x)
                env.set_value(f"p2_{i+1}_y", y)
    else:
        raise ValueError(f"Invalid environment name, got '{env_name}'")

def isdone_defensezone(state):
    t1 = state.team1
    t2 = state.team2

    if state.puck.y >= 100:
        return True

    if t2.stats.score > t2.last_stats.score:
        return True

    if state.time < 200:
        return True

def rf_defensezone(state):
    t1 = state.team1
    t2 = state.team2

    tracker = getattr(state, "t1", None)

    # Track short-lived possessions so dumping the puck is penalized
    carry_state = getattr(state, "_defensezone_carry", None)
    if carry_state is None:
        carry_state = {
            "last_player_haspuck": t1.player_haspuck,
            "possession_frames": 1 if t1.player_haspuck else 0,
            "carry_start_y": state.puck.y if t1.player_haspuck else None,
            "last_carry_y": state.puck.y if t1.player_haspuck else None,
        }
        setattr(state, "_defensezone_carry", carry_state)

    def _closest_team_distance(team, puck):
        members = []
        goalie = getattr(team, "goalie", None)
        if goalie is not None:
            members.append(goalie)
        members.extend(getattr(team, "players", []))
        distances = [
            math.hypot(player.x - puck.x, player.y - puck.y)
            for player in members
            if player is not None
        ]
        return min(distances) if distances else None

    dist_to_puck = getattr(tracker, "distToPuck", None)
    if dist_to_puck is None:
        dist_to_puck = _closest_team_distance(t1, state.puck)

    last_dist_to_puck = getattr(tracker, "last_distToPuck", None)
    if last_dist_to_puck is None:
        last_dist_to_puck = dist_to_puck

    reward = 0.0

    if not t1.player_haspuck:
        if dist_to_puck is not None and last_dist_to_puck is not None:
            if dist_to_puck < last_dist_to_puck:
                reward += 1 - (dist_to_puck / 200.0) ** 0.5
            else:
                reward -= 0.1
        else:
            reward -= 0.05

        if carry_state.get("last_player_haspuck", False):
            possession_frames = carry_state.get("possession_frames", 0)
            last_carry_y = carry_state.get("last_carry_y")
            if last_carry_y is None:
                last_carry_y = state.puck.y
            carry_start_y = carry_state.get("carry_start_y")
            if carry_start_y is None:
                carry_start_y = last_carry_y
            forward_gain = last_carry_y - carry_start_y
            launched_forward = state.puck.vy > 12 and state.puck.y > last_carry_y
            still_inside_zone = last_carry_y < GameConsts.DEFENSEZONE_POS_Y
            if possession_frames < 20 and forward_gain < 25 and launched_forward and still_inside_zone:
                reward -= 1.0

        carry_state["possession_frames"] = 0
        carry_state["carry_start_y"] = None
        carry_state["last_carry_y"] = None
    else:
        if not carry_state.get("last_player_haspuck", False):
            carry_state["carry_start_y"] = state.puck.y
            carry_state["possession_frames"] = 0

        carry_state["possession_frames"] = carry_state.get("possession_frames", 0) + 1
        carry_state["last_carry_y"] = state.puck.y

        progress = 0.0
        if carry_state.get("carry_start_y") is not None:
            progress = state.puck.y - carry_state["carry_start_y"]
        if progress > 0:
            reward += min(0.3, 0.02 * (progress / 5.0))

        reward += 0.1

        if state.action[5] and state.puck.y < GameConsts.DEFENSEZONE_POS_Y + 15:
            reward -= 0.4

        if state.puck.y >= GameConsts.ATACKZONE_POS_Y:
            start_y = carry_state.get("carry_start_y")
            attackzone_requires_carry = (
                progress >= 20
                or carry_state["possession_frames"] >= 20
                or (start_y is not None and start_y >= GameConsts.DEFENSEZONE_POS_Y - 15)
            )
            if attackzone_requires_carry:
                reward += 1.0
            else:
                reward -= 0.3

    if t1.stats.bodychecks > t1.last_stats.bodychecks:
        reward += 1.0

    if t1.stats.passing > t1.last_stats.passing:
        reward += 1.0

    if not t1.player_haspuck:
        first_player = t1.players[0] if t1.players else None
        if first_player is not None and first_player.y > -80:
            reward -= 1.0
        if state.puck.y > -80:
            reward -= 1.0

    if t1.goalie_haspuck:
        reward -= 1.0

    if t2.stats.score > t2.last_stats.score:
        reward -= 1.0

    if t2.stats.shots > t2.last_stats.shots:
        reward -= 0.1

    carry_state["last_player_haspuck"] = t1.player_haspuck

    return reward

# =====================================================================
# Passing
# =====================================================================
def isdone_passing(state):
    t1 = state.team1
    t2 = state.team2

    if state.puck.y < 100:
        return True

    if state.time < 100:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    return False

def rf_passing(state):
    t1 = state.team1
    t2 = state.team2

    pass_state = getattr(state, "_passing_attempt", None)
    if pass_state is None:
        pass_state = {
            "pending_attempt": False,
            "released_puck": False,
            "frames_since_attempt": 0,
            "last_pass_button": False,
        }
        setattr(state, "_passing_attempt", pass_state)

    rew = 0.0

    pass_button_pressed = bool(state.action[4])
    new_pass_press = pass_button_pressed and not pass_state["last_pass_button"]

    if new_pass_press and t1.player_haspuck:
        pass_state["pending_attempt"] = True
        pass_state["released_puck"] = False
        pass_state["frames_since_attempt"] = 0

    if pass_state["pending_attempt"]:
        pass_state["frames_since_attempt"] += 1
        if not t1.player_haspuck and not t2.player_haspuck and not t2.goalie_haspuck:
            pass_state["released_puck"] = True

    if t2.player_haspuck or t2.goalie_haspuck:
        rew = -0.1
        pass_state["pending_attempt"] = False
        pass_state["released_puck"] = False
        pass_state["frames_since_attempt"] = 0

    #if state.puck.y < 100:
    #    rew = -1.0

    #if t1.stats.score > t1.last_stats.score:
    #    rew = -1.0

    if t1.stats.passing > t1.last_stats.passing:
        rew = 1.0
        pass_state["pending_attempt"] = False
        pass_state["released_puck"] = False
        pass_state["frames_since_attempt"] = 0
    elif pass_state["pending_attempt"] and pass_state["released_puck"] and t1.player_haspuck:
        rew += 0.05
        pass_state["pending_attempt"] = False
        pass_state["released_puck"] = False
        pass_state["frames_since_attempt"] = 0
    elif pass_state["pending_attempt"] and pass_state["frames_since_attempt"] >= 8:
        pass_state["pending_attempt"] = False
        pass_state["released_puck"] = False
        pass_state["frames_since_attempt"] = 0

    #rew -= 0.001

    pass_state["last_pass_button"] = pass_button_pressed

    return rew

# =====================================================================
# Self Play
# =====================================================================
def isdone_selfplay(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score:
        return True

    if state.time < 100:
        return True

    return False

def init_selfplay(env, env_name):
    """Same random (or attack-zone) init as normal."""
    init_general(env, env_name)      # or init_attackzone(...)

def rf_selfplay(state):
    """
    Zero-sum reward wrapper around any existing reward.
    The wrapper decides which side is 'active' and flips the sign.
    """
    # base reward computed from Team-1 perspective
    base = rf_general(state)         # or rf_scoregoal_cc, etc.
    # wrapper will negate if training Team-2
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

# =====================================================================
# Register Functions
# =====================================================================
_reward_function_map = {
    "GetPuck": (init_getpuck, rf_getpuck, isdone_getpuck, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "GetPuckAZ": (init_getpuck_az, rf_getpuck, isdone_getpuck_az, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "GetPuckNZ": (init_getpuck_nz, rf_getpuck, isdone_getpuck_nz, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "GetPuckDZ": (init_getpuck_dz, rf_getpuck, isdone_getpuck_dz, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "ScoreGoalCC": (init_attackzone, rf_scoregoal_cc, isdone_scoregoal_cc, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons, input_overide_empty),
    "CrossCreaseV2": (init_attackzone, rf_crosscrease_v2, isdone_crosscrease_v2, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "ScoreGoalOT": (init_attackzone, rf_scoregoal_ot, isdone_scoregoal_ot, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "ScoreGoal": (init_attackzone, rf_scoregoal, isdone_scoregoal, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "ScoreGoalV2": (init_attackzone, rf_scoregoal_v2, isdone_scoregoal_v2, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "KeepPuck": (init_keeppuck, rf_keeppuck, isdone_keeppuck, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "DefenseZone": (init_defensezone, rf_defensezone, isdone_defensezone, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "Passing": (init_attackzone, rf_passing, isdone_passing, init_model_rel_dist_buttons_v2, set_model_input_rel_dist_buttons_v2, input_overide_no_shoot),
    "General": (init_general, rf_general, isdone_general, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons_v2, input_overide_empty),
    "SelfPlay": (init_selfplay, rf_selfplay, isdone_selfplay, init_model_invariant, set_model_input_invariant, input_overide_empty),
    "SelfPlayOffenseFinetune": (init_selfplay_offense, rf_selfplay_offense, isdone_selfplay_offense, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons, input_overide_empty),
    "SelfPlayDefenseFinetune": (init_selfplay_defense, rf_selfplay_defense, isdone_selfplay_defense, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons, input_overide_empty),
}

def register_functions(name: str) -> Tuple[Callable, Callable, Callable]:
    if name not in _reward_function_map:
        raise ValueError(f"Unsupported Reward Function: {name}")
    return _reward_function_map[name]
