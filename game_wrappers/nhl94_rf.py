"""
NHL94 Reward Functions
"""

import random
from typing import Tuple, Callable
from game_wrappers.nhl94_const import GameConsts

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
    y = (random.random() * GameConsts.SPAWNABLE_ZONE_HEIGHT) + 100

    #print(x,y)
    return x, y

def RandomPosDefenseZone():
    x = (random.random() - 0.5) * GameConsts.SPAWNABLE_AREA_WIDTH
    y = (random.random() * GameConsts.SPAWNABLE_ZONE_HEIGHT) - 220

    #print(x,y)
    return x, y

# =====================================================================
# General
# =====================================================================
def init_general(state):
    return

def isdone_general(state):
    if state.time < 10:
        return True

    return False

def rf_general(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    if t1.stats.score > t1.last_stats.score:
        rew = 1.0

    if state.t2.stats.score > t2.last_stats.score:
        rew = -1.0

    return rew

# =====================================================================
# ScoreGoal
# =====================================================================
def init_scoregoal(env):
    x, y = RandomPosAttackZone()
    env.set_value("p2_x", x)
    env.set_value("p2_y", y)

    x, y = RandomPosAttackZone()
    env.set_value("p1_x", x)
    env.set_value("p1_y", y)

def isdone_scoregoal(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score: #or self.game_state.p1_shots > self.game_state.last_p1_shots:
        return True

    #if state.p2_haspuck or state.g2_haspuck:
    #    return True

    #if state.puck_y < 100:
    #    return True

    if state.time < 100:
        return True

    return False

def rf_scoregoal(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    if t2.player_haspuck or t2.goalie_haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    if t1.stats.score > t1.last_stats.score:
        rew = 1.0

    # reward scoring opportunities
    if t1.player_haspuck and t1.players[0].y < GameConsts.CREASE_UPPER_BOUND and t1.players[0].y  > GameConsts.CREASE_LOWER_BOUND:
        if t1.players[0].vx >= GameConsts.CREASE_MIN_VEL or t1.players[0].vx <= -GameConsts.CREASE_MIN_VEL:
            rew = 0.2
            if state.puck.x > -GameConsts.CREASE_MAX_X and state.puck.x < GameConsts.CREASE_MAX_X:
                if abs(state.puck.x - t2.goalie.x) > GameConsts.CREASE_MIN_GOALIE_PUCK_DIST_X:
                    rew = 1.0
                else:
                    rew = 0.5

    return rew

# =====================================================================
# ScoreGoal02
# =====================================================================
def init_scoregoal02(env):
    x, y = RandomPosAttackZone()
    env.set_value("p2_x", x)
    env.set_value("p2_y", y)

    x, y = RandomPosAttackZone()
    env.set_value("p1_x", x)
    env.set_value("p1_y", y)

def isdone_scoregoal02(state):

    if state.stats.score > state.last_stats.score:
        return True

    #if state.p1_shots > state.last_p1_shots:
    #    init_scoregoal02(env)

    #if state.p2_haspuck or state.g2_haspuck:
    #    return True

    #if state.puck_y < 100:
    #    return True

    if state.time < 250:
        return True

    return False

def rf_scoregoal02(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    if t2.haspuck or t2.goalie.haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    if state.t1.stats.score > state.t1.last_stats.score:
        rew = 1.0

    # reward scoring opportunities
    if t1.haspuck and t1.players[0].y < GameConsts.CREASE_UPPER_BOUND and t1.players[0].y > GameConsts.CREASE_LOWER_BOUND:
        if t1.players[0].vx >= GameConsts.CREASE_MIN_VEL or t1.players[0].vx <= -GameConsts.CREASE_MIN_VEL:
            rew = 0.2
            if state.puck.x > -GameConsts.CREASE_MAX_X and state.puck.x < GameConsts.CREASE_MAX_X:
                if abs(state.puck.x - t2.goalie.x) > GameConsts.CREASE_MIN_GOALIE_PUCK_DIST_X:
                    rew = 1.0
                else:
                    rew = 0.5

    return rew

# =====================================================================
# KeepPuck
# =====================================================================
def init_keeppuck(env):
    #x, y = self.RandomPos()
    #self.env.set_value("rpuck_x", x)
    #self.env.set_value("rpuck_y", y)

    x, y = RandomPos()
    env.set_value("p2_x", x)
    env.set_value("p2_y", y)

    x, y = RandomPos()
    env.set_value("p1_x", x)
    env.set_value("p1_y", y)

def isdone_keeppuck(state):
    t1 = state.team1
    t2 = state.team2

    if state.t1.haspuck == False:
        return True

def rf_keeppuck(state):

    rew = 1.0
    if not state.t1.haspuck:
        rew = -1.0

    return rew

# =====================================================================
# GetPuck
# =====================================================================
def init_getpuck(env):
    #x, y = self.RandomPos()
    #self.env.set_value("rpuck_x", x)
    #self.env.set_value("rpuck_y", y)

    x, y = RandomPos()
    env.set_value("p2_x", x)
    env.set_value("p2_y", y)

    x, y = RandomPos()
    env.set_value("p1_x", x)
    env.set_value("p1_y", y)

def isdone_getpuck(state):
    #if state.player_haspuck == True:
        #print('TERMINATED: GOT PUCK: (%d,%d) (%d,%d)' % (info.get('p1_x'), info.get('p1_y'), fullstar_x, fullstar_y))
    #    return True
    if state.time < 100:
        return True

def rf_getpuck(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0

    scaled_dist = state.t1.distToPuck / 200.0

    if t1.haspuck == False:
        if t1.distToPuck < t1.last_distToPuck:
            #rew = 1.0 / (1.0 + scaled_dist)
            rew = 1 - (t1.distToPuck / 200.0)**0.5
            #print(state.distToPuck, rew)
        else:
            rew = -0.1
    else:
        rew = 1

    #if state.p1_bodychecks > state.last_p1_bodychecks:
    #    rew = 0.5

    if t1.goalie_haspuck:
        rew = -1

    if t2.stats.score > t2.last_stats.score:
        rew = -1.0

    if t1.stats.shots > t1.last_stats.shots:
        rew = -1.0

    #if state.time < 200:
    #    rew = -1

    return rew

# =====================================================================
# DefenseZone
# =====================================================================
def init_defensezone(env):
    #x, y = self.RandomPos()
    #self.env.set_value("rpuck_x", x)
    #self.env.set_value("rpuck_y", y)

    x, y = RandomPosDefenseZone()
    env.set_value("p2_x", x)
    env.set_value("p2_y", y)

    x, y = RandomPosDefenseZone()
    env.set_value("p1_x", x)
    env.set_value("p1_y", y)

def isdone_defensezone(state):
    #if state.player_haspuck and state.puck_y > - 80:
        #print('TERMINATED: GOT PUCK: (%d,%d) (%d,%d)' % (info.get('p1_x'), info.get('p1_y'), fullstar_x, fullstar_y))
    #    return True

    #if state.p2_score > state.last_p2_score:
    #    return True

    if state.time < 100:
        return True

def rf_defensezone(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0

    if state.player_haspuck == False:
        if state.distToPuck < state.last_dist:
            #rew = 1.0 / (1.0 + scaled_dist)
            rew = 1 - (state.distToPuck / 200.0)**0.5
            #print(state.distToPuck, rew)
        else:
            rew = -0.1
    else:
        rew = 1


    if state.state.stats.bodychecks > state.last_bodychecks:
        rew = 1.0

    if state.t1.stats.passing > state.t1.stats.last_passing:
        rew = 1.0

    if not state.t1.player_haspuck:
        if state.players[0].y > -80:
            rew = -1.0
        if state.puck.y > -80:
            rew = -1.0

    if state.t1.goalie_haspuck:
        rew = -1.0

    if state.t2.stats.score > state.t2.last_stats.score:
        rew = -1.0

    if state.t2.stats.shots > state.t2.last_stats.shots:
        rew = -1.0

    #if state.time < 200:
    #    rew = -1

    return rew

# =====================================================================
# Passing
# =====================================================================
def init_passing(env):
    x, y = RandomPosAttackZone()
    env.set_value("p2_x", x)
    env.set_value("p2_y", y)

    x, y = RandomPosAttackZone()
    env.set_value("p1_x", x)
    env.set_value("p1_y", y)

def isdone_passing(state):
    t1 = state.team1
    t2 = state.team2

    if state.puck_y < 100:
        return True

    if state.p2_haspuck or state.g2_haspuck:
        return True

    if state.time < 100:
        return True

    return False

def rf_passing(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    if state.t2.player_haspuck or state.t2.goalie_haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    if state.t1.stats.passing > state.t1.last_stats.passing:
        rew = 1.0

    return rew

# =====================================================================
# Register Functions
# =====================================================================
_reward_function_map = {
    "GetPuck": (init_getpuck, rf_getpuck, isdone_getpuck),
    "ScoreGoal": (init_scoregoal, rf_scoregoal, isdone_scoregoal),
    "ScoreGoal02": (init_scoregoal02, rf_scoregoal02, isdone_scoregoal02),
    "KeepPuck": (init_keeppuck, rf_keeppuck, isdone_keeppuck),
    "DefenseZone": (init_defensezone, rf_defensezone, isdone_defensezone),
    "Passing": (init_passing, rf_passing, isdone_passing),
    "General": (init_general, rf_general, isdone_general),
}

def register_functions(name: str) -> Tuple[Callable, Callable, Callable]:
    if name not in _reward_function_map:
        raise ValueError(f"Unsupported Reward Function: {name}")
    return _reward_function_map[name]
