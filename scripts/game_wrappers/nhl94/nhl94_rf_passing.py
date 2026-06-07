"""
NHL94 passing reward functions.
"""


def isdone_passing(state):
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

    pass_state["last_pass_button"] = pass_button_pressed

    return rew