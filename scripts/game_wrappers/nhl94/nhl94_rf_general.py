"""
NHL94 general reward functions.
"""

from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_rf import init_general
from game_wrappers.nhl94.nhl94_rf_scoregoal import (
    _ensure_attack_tracker,
    _reset_attack_tracker,
    _scoregoal_v2_attack_reward,
)


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

    if t1.stats.score > t1.last_stats.score:
        return 1.0
    if t2.stats.score > t2.last_stats.score:
        return -1.0

    if t2.stats.shots > t2.last_stats.shots:
        return -0.5

    return rew


def init_general_v2(env, env_name):
    init_general(env, env_name)


def isdone_general_v2(state):
    return isdone_general(state)


def rf_general_v2(state):
    t1 = state.team1
    t2 = state.team2
    puck_y = state.puck.y
    team1_has_control = bool(t1.player_haspuck or t1.goalie_haspuck)
    team2_has_control = bool(t2.player_haspuck or t2.goalie_haspuck)

    tracker = getattr(state, "_general_v2_tracker", None)
    if tracker is None:
        tracker = {
            "last_zone": None,
            "frames_in_zone": 0,
            "last_puck_y": puck_y,
            "had_team1_control": team1_has_control,
            "had_team2_control": team2_has_control,
        }
        setattr(state, "_general_v2_tracker", tracker)

    if puck_y <= GameConsts.DEFENSEZONE_POS_Y:
        zone = "defense"
    elif puck_y >= GameConsts.ATACKZONE_POS_Y:
        zone = "attack"
    else:
        zone = "neutral"

    last_zone = tracker["last_zone"]
    zone_transition = None
    if last_zone is not None and last_zone != zone:
        zone_transition = (last_zone, zone)

    rew = 0.0

    if t1.stats.score > t1.last_stats.score:
        return 1.0
    if t2.stats.score > t2.last_stats.score:
        return -1.0

    puck_delta_y = puck_y - tracker["last_puck_y"]
    controlled_forward_delta = max(0.0, puck_delta_y) if team1_has_control else 0.0
    controlled_backward_delta = min(0.0, puck_delta_y) if team1_has_control else 0.0

    if zone_transition == ("defense", "neutral") and team1_has_control:
        rew += 0.15
    elif zone_transition == ("neutral", "attack") and team1_has_control:
        rew += 0.10
    elif zone_transition == ("attack", "neutral"):
        if tracker["had_team1_control"] or team1_has_control:
            rew -= 0.12
    elif zone_transition == ("neutral", "defense"):
        if tracker["had_team2_control"] or team2_has_control:
            rew -= 0.12
    elif zone_transition == ("defense", "attack") and team1_has_control:
        rew += 0.25
    elif zone_transition == ("attack", "defense"):
        if tracker["had_team1_control"] or team1_has_control:
            rew -= 0.30

    if zone == "defense":
        rew -= 0.01

        if t1.player_haspuck:
            rew += min(controlled_forward_delta * 0.003, 0.025)
            rew += max(controlled_backward_delta * 0.004, -0.03)
            rew -= 0.003
            if t1.stats.passing > t1.last_stats.passing:
                rew += 0.02
            if state.action[5] and puck_y < GameConsts.DEFENSEZONE_POS_Y + 15:
                rew -= 0.2
        elif t1.goalie_haspuck:
            rew -= 0.08
        elif team2_has_control:
            rew -= 0.05

        if t2.stats.shots > t2.last_stats.shots:
            rew -= 0.4

        if t1.stats.bodychecks > t1.last_stats.bodychecks:
            rew += 0.12

        if t1.stats.shots > t1.last_stats.shots:
            rew -= 0.2

    elif zone == "attack":
        attack_tracker = _ensure_attack_tracker(state, "_general_v2_attack_tracker")
        rew += _scoregoal_v2_attack_reward(state, attack_tracker, turnover_penalty=-0.15)

    else:
        if t1.player_haspuck:
            rew += min(controlled_forward_delta * 0.0025, 0.02)
            rew += max(controlled_backward_delta * 0.003, -0.025)
            rew -= 0.002
            if t1.stats.passing > t1.last_stats.passing:
                rew += 0.03
        elif team2_has_control:
            rew -= 0.01

        if t2.stats.shots > t2.last_stats.shots:
            rew -= 0.2

    if zone != "attack":
        attack_tracker = getattr(state, "_general_v2_attack_tracker", None)
        if attack_tracker is not None:
            _reset_attack_tracker(attack_tracker, state)

    if t1.stats.bodychecks > t1.last_stats.bodychecks:
        rew += 0.05

    if zone != last_zone:
        tracker["frames_in_zone"] = 0
    else:
        tracker["frames_in_zone"] += 1

    tracker["last_zone"] = zone
    tracker["last_puck_y"] = puck_y
    tracker["had_team1_control"] = team1_has_control
    tracker["had_team2_control"] = team2_has_control

    return rew