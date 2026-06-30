"""
NHL94 score-goal style reward functions.
"""

import math

from game_wrappers.nhl94.nhl94_const import GameConsts


def isdone_scoregoal_cc(state):
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


def rf_scoregoal_cc(state):
    t1 = state.team1
    t2 = state.team2
    rew = 0.0

    if t2.player_haspuck or t2.goalie_haspuck:
        return -0.5

    if state.puck.y < 100:
        return -0.5

    controlled_player = t1.players[t1.control - 1] if t1.control > 0 else t1.goalie
    in_crease = (
        controlled_player.y < GameConsts.CREASE_UPPER_BOUND
        and controlled_player.y > GameConsts.CREASE_LOWER_BOUND
    )

    if in_crease:
        if abs(controlled_player.x) < GameConsts.CREASE_MAX_X * 3:
            rew += 0.1

        if abs(controlled_player.x) < GameConsts.CREASE_MAX_X:
            rew += 0.4

        if abs(controlled_player.vx) > 15:
            rew = 0.5

    if state.action[5] == 1 and in_crease:
        if abs(controlled_player.x - t2.goalie.x) > GameConsts.CREASE_MIN_GOALIE_PUCK_DIST_X:
            rew += 1.0
        else:
            rew += 0.2

    if t1.stats.passing > t1.last_stats.passing:
        rew += 0.5

    if t1.stats.score > 0:
        return 1.0

    return rew


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
    controlled_player = t1.get_controlled_player()
    opponent_goalie = t2.goalie
    rew = 0.0

    if controlled_player.one_timer_lane_good:
        rew = max(rew, 1.0)

    if controlled_player.clear_shot_lane:
        rew = max(rew, 0.1)

    if controlled_player.open_net_shot:
        rew = max(rew, 1.0)

    if opponent_goalie.is_dive or opponent_goalie.is_pad_stack:
        rew = max(rew, 0.2)

    if state.engine.goalie_box_small:
        rew = max(rew, 0.2)

    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew = max(rew, 0.1)

    if t1.stats.score > t1.last_stats.score:
        rew = 1.0

    return rew


def isdone_scoregoal(state):
    t1 = state.team1

    if t1.stats.score > t1.last_stats.score:
        return True

    if state.puck.y < 100:
        return True

    if state.time < 100:
        return True
    

    return False


def rf_scoregoal(state):
    t1 = state.team1
    rew = 0.0

    if t1.stats.passing > t1.last_stats.passing:
        rew = 0.1

    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew = 0.1

    if t1.stats.score > t1.last_stats.score:
        rew = 1.0

    return rew




def _ensure_attack_tracker(state, attr_name):
    tracker = getattr(state, attr_name, None)
    if tracker is None:
        tracker = {
            "possession_frames": 0,
            "stall_frames": 0,
            "shot_mode_frames": 0,
            "best_attack_y": state.puck.y,
            "last_puck_x": state.puck.x,
            "last_puck_y": state.puck.y,
            "prev_shot_mode_active": bool(state.engine.shot_mode_active),
            "prev_shot_taken": bool(state.engine.shot_taken),
            "prev_top_shelf": bool(state.engine.in_close_top_shelf),
            "prev_in_danger_area": False,
            "prev_team1_haspuck": bool(state.team1.player_haspuck or state.team1.goalie_haspuck),
            "prev_team2_haspuck": bool(state.team2.player_haspuck or state.team2.goalie_haspuck),
            "shot_taken_once": False,
        }
        setattr(state, attr_name, tracker)
    return tracker


def _reset_attack_tracker(tracker, state):
    tracker["possession_frames"] = 0
    tracker["stall_frames"] = 0
    tracker["shot_mode_frames"] = 0
    tracker["best_attack_y"] = max(GameConsts.ATACKZONE_POS_Y, state.puck.y)
    tracker["last_puck_x"] = state.puck.x
    tracker["last_puck_y"] = state.puck.y
    tracker["prev_shot_mode_active"] = bool(state.engine.shot_mode_active)
    tracker["prev_shot_taken"] = bool(state.engine.shot_taken)
    tracker["prev_top_shelf"] = bool(state.engine.in_close_top_shelf)
    tracker["prev_in_danger_area"] = False
    tracker["prev_team1_haspuck"] = bool(state.team1.player_haspuck or state.team1.goalie_haspuck)
    tracker["prev_team2_haspuck"] = bool(state.team2.player_haspuck or state.team2.goalie_haspuck)
    tracker["shot_taken_once"] = False


def _scoregoal_v2_attack_reward(state, tracker, turnover_penalty, exit_penalty=None):
    t1 = state.team1
    t2 = state.team2
    engine = state.engine

    if t1.stats.score > t1.last_stats.score:
        return 1.0

    if t2.player_haspuck or t2.goalie_haspuck:
        _reset_attack_tracker(tracker, state)
        tracker["prev_team1_haspuck"] = False
        tracker["prev_team2_haspuck"] = True
        return turnover_penalty

    if exit_penalty is not None and state.puck.y < GameConsts.ATACKZONE_POS_Y:
        exited_after_team1_control = tracker["prev_team1_haspuck"] and not tracker["prev_team2_haspuck"]
        _reset_attack_tracker(tracker, state)
        tracker["prev_team1_haspuck"] = bool(t1.player_haspuck or t1.goalie_haspuck)
        tracker["prev_team2_haspuck"] = bool(t2.player_haspuck or t2.goalie_haspuck)
        return exit_penalty[0] if exited_after_team1_control else exit_penalty[1]

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
    deep_corner_trap = controlled_player.y > 185 and abs(controlled_player.x) > 72
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
            rew -= 0.08
            if deep_corner_trap:
                rew -= 0.08

        lateral_escape = abs(tracker["last_puck_x"]) - abs(state.puck.x)
        if state.puck.y > 135 and lateral_escape > 0:
            rew += min(0.14, lateral_escape * 0.012)
            if corner_trap:
                rew += min(0.06, lateral_escape * 0.008)

        if puck_step < 2.0 and control_speed < 4.0 and not engine.shot_mode_active and not state.action[4] and not state.action[5]:
            tracker["stall_frames"] += 1
        else:
            tracker["stall_frames"] = max(0, tracker["stall_frames"] - 2)

        if tracker["stall_frames"] > 15:
            rew -= min((tracker["stall_frames"] - 15) * 0.004, 0.12)
            if corner_trap:
                rew -= min((tracker["stall_frames"] - 15) * 0.006, 0.12)
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


def isdone_scoregoal_v2(state):
    t1 = state.team1

    if t1.stats.score > t1.last_stats.score:
        return True

    if state.puck.y < GameConsts.ATACKZONE_POS_Y:
        return True

    if state.time < 100:
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
    tracker = _ensure_attack_tracker(state, "_scoregoal_v2_tracker")
    return _scoregoal_v2_attack_reward(state, tracker, turnover_penalty=-0.05, exit_penalty=(-1.0, -0.5))


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
            "prev_shot_mode_active": bool(engine.shot_mode_active),
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

    tracker.setdefault("prev_shot_mode_active", bool(engine.shot_mode_active))

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
    shot_mode_started = bool(engine.shot_mode_active) and not tracker["prev_shot_mode_active"]
    shot_taken_now = bool(engine.shot_taken) and not tracker["prev_shot_taken"]
    top_shelf_started = bool(engine.in_close_top_shelf) and not tracker["prev_top_shelf"]

    near_net = controlled_player.y > (GameConsts.CREASE_LOWER_BOUND - 10)
    side_setup = near_net and abs(controlled_player.x) > GameConsts.CREASE_MAX_X and abs(controlled_player.x) < 75
    deep_side_setup = side_setup and abs(controlled_player.x) > (GameConsts.CREASE_MAX_X + 10)
    danger_receiver_area = controlled_player.y > GameConsts.CREASE_LOWER_BOUND and abs(controlled_player.x) < GameConsts.CREASE_MAX_X * 2
    corner_trap = controlled_player.y > 150 and abs(controlled_player.x) > 65
    deep_corner_trap = controlled_player.y > 185 and abs(controlled_player.x) > 78

    rew = 0.0

    if t1.player_haspuck:
        tracker["possession_frames"] += 1
        tracker["best_attack_y"] = max(tracker["best_attack_y"], state.puck.y)

        rew -= 0.004

        if tracker["possession_frames"] > 70:
            rew -= min((tracker["possession_frames"] - 70) * 0.0015, 0.08)

        if corner_trap:
            rew -= 0.08
            if deep_corner_trap:
                rew -= 0.10

        lateral_escape = abs(tracker["last_puck_x"]) - abs(state.puck.x)
        if state.puck.y > 145 and lateral_escape > 0:
            rew += min(0.16, lateral_escape * 0.014)
            if corner_trap:
                rew += min(0.08, lateral_escape * 0.01)

        if puck_step < 2.0 and control_speed < 4.0 and not state.action[4] and not state.action[5]:
            tracker["stall_frames"] += 1
        else:
            tracker["stall_frames"] = max(0, tracker["stall_frames"] - 2)

        if tracker["stall_frames"] > 12:
            rew -= min((tracker["stall_frames"] - 12) * 0.005, 0.14)
            if corner_trap:
                rew -= min((tracker["stall_frames"] - 12) * 0.006, 0.12)
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

    if shot_mode_started and near_net and (goalie_committed or tracker["commit_window"] > 0):
        rew += 0.25
        if controlled_player.open_net_shot:
            rew += 0.15

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

    tracker["prev_shot_mode_active"] = bool(engine.shot_mode_active)
    tracker["prev_shot_taken"] = bool(engine.shot_taken)
    tracker["prev_top_shelf"] = bool(engine.in_close_top_shelf)
    tracker["last_puck_x"] = state.puck.x
    tracker["last_puck_y"] = state.puck.y

    return rew






def isdone_scoregoal_v4(state):
    return isdone_scoregoal(state)


def _ensure_scoregoal_v4_tracker(state):
    tracker = getattr(state, "_scoregoal_v4_tracker", None)
    if tracker is None:
        tracker = {
            "prev_good_one_timer_lane": False,
        }
        setattr(state, "_scoregoal_v4_tracker", tracker)
    return tracker


def rf_scoregoal_v4(state):
    t1 = state.team1
    tracker = _ensure_scoregoal_v4_tracker(state)
    rew = 0.0

    has_good_one_timer_lane = t1.player_haspuck and any(
        player.one_timer_lane_good for player in t1.players
    )

    if has_good_one_timer_lane and not tracker["prev_good_one_timer_lane"]:
        rew += 0.05

    if t1.stats.passing > t1.last_stats.passing:
        rew += 0.05
        if tracker["prev_good_one_timer_lane"]:
            rew += 0.15

    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew += 0.35

    if t1.stats.score > t1.last_stats.score:
        rew = 1.0

    tracker["prev_good_one_timer_lane"] = has_good_one_timer_lane

    return rew
