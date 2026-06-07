"""
NHL94 post-play reward functions.
"""


def init_postplay(env, env_name):
    return None


def isdone_postplay(state):
    if state.time < 10:
        return True

    return False


def rf_postplay(state):
    return 0.0