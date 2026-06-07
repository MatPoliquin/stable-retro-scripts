"""
NHL94 defense-zone reward functions.
"""

from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_rf import SampleRandomPosDefenseZone


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
        for i in range(5):
            x, y = SampleRandomPosDefenseZone(env)
            if i == 0:
                env.set_value("p1_x", x)
                env.set_value("p1_y", y)
            else:
                env.set_value(f"p1_{i+1}_x", x)
                env.set_value(f"p1_{i+1}_y", y)

        for i in range(5):
            x, y = SampleRandomPosDefenseZone(env)
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
    team1_has_control = bool(t1.player_haspuck or t1.goalie_haspuck)

    if state.puck.y >= GameConsts.ATACKZONE_POS_Y and team1_has_control:
        return True

    if t2.stats.score > t2.last_stats.score:
        return True

    if state.time < 200:
        return True


def rf_defensezone(state):
    t1 = state.team1
    t2 = state.team2

    tracker = getattr(state, "_defensezone_tracker", None)
    if tracker is None:
        tracker = {
            "last_puck_y": state.puck.y,
        }
        setattr(state, "_defensezone_tracker", tracker)

    team1_has_control = bool(t1.player_haspuck or t1.goalie_haspuck)
    puck_delta_y = state.puck.y - tracker["last_puck_y"]

    reward = 0.0

    if team1_has_control and puck_delta_y > 0:
        reward += min(0.25, puck_delta_y * 0.01)

    if state.puck.y >= GameConsts.ATACKZONE_POS_Y and team1_has_control:
        reward += 1.0

    if t1.stats.shots > t1.last_stats.shots:
        reward -= 0.5

    if t2.stats.shots > t2.last_stats.shots:
        reward -= 0.75

    if t1.stats.score > t1.last_stats.score:
        reward -= 1.0

    if t2.stats.score > t2.last_stats.score:
        reward -= 1.0

    tracker["last_puck_y"] = state.puck.y

    return reward