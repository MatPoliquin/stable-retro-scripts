"""
NHL94 attack-zone lost-puck reward functions.
"""

from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_rf import SampleRandomPosAttackZone, SetRandomSkaterPositions


def init_az_lostpuck(env, env_name=None):
    if env_name is None:
        env_name = 'NHL941on1-Genesis-v0'

    SetRandomSkaterPositions(env, env_name, SampleRandomPosAttackZone)


def _team_has_control(team):
    return bool(team.player_haspuck or team.goalie_haspuck)


def _team2_controls_in_team1_defense_zone(state):
    return _team_has_control(state.team2) and state.puck.y <= GameConsts.DEFENSEZONE_POS_Y


def isdone_az_lostpuck(state):
    if _team_has_control(state.team1):
        return True

    if _team2_controls_in_team1_defense_zone(state):
        return True

    return False


def rf_az_lostpuck(state):
    if _team_has_control(state.team1):
        return 1.0

    if _team2_controls_in_team1_defense_zone(state):
        return -1.0

    return 0.0