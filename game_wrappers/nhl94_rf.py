"""
NHL94 Reward Functions
"""

import random
from game_wrappers.nhl94_gamestate import NHL94GameState


# =====================================================================
# Common functions
# =====================================================================
def RandomPos():
    x = (random.random() - 0.5) * 235
    y = (random.random() - 0.5) * 460

    #print(x,y)
    return x, y

def RandomPosAttackZone():
    x = (random.random() - 0.5) * 235
    y = (random.random() * 140) + 100

    #print(x,y)
    return x, y

def RandomPosDefenseZone():
    x = (random.random() - 0.5) * 235
    y = (random.random() * 140) - 220

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
    rew = 0.0

    if state.p1_score > state.last_p1_score:
        rew = 1.0

    if state.p2_score > state.last_p2_score:
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
    if state.p1_score > state.last_p1_score: #or self.game_state.p1_shots > self.game_state.last_p1_shots:
        return True

    #if state.p2_haspuck or state.g2_haspuck:
    #    return True

    #if state.puck_y < 100:
    #    return True

    if state.time < 100:
        return True

    return False

def rf_scoregoal(state):

    rew = 0.0

    if state.p2_haspuck or state.g2_haspuck:
        rew = -1.0

    if state.puck_y < 100:
        rew = -1.0

    if state.p1_score > state.last_p1_score:
        rew = 1.0

    # reward scoring opportunities
    if state.player_haspuck and state.p1_y < 230 and state.p1_y > 210:
        if state.p1_vel_x >= 30 or state.p1_vel_x <= -30:
            rew = 0.2
            if state.puck_x > -23 and state.puck_x < 23:
                if abs(state.puck_x - state.g2_x) > 7:
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

    if state.p1_score > state.last_p1_score:
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

    rew = 0.0

    if state.p2_haspuck or state.g2_haspuck:
        rew = -1.0

    if state.puck_y < 100:
        rew = -1.0

    if state.p1_score > state.last_p1_score:
        rew = 1.0

    # reward scoring opportunities
    if state.player_haspuck and state.p1_y < 230 and state.p1_y > 210:
        if state.p1_vel_x >= 30 or state.p1_vel_x <= -30:
            rew = 0.2
            if state.puck_x > -23 and state.puck_x < 23:
                if state.p1_shots > state.last_p1_shots:
                    rew = 1.0

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
    if state.player_haspuck == False:
        return True

def rf_keeppuck(state):

    rew = 1.0
    if not state.player_haspuck:
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
    rew = 0

    scaled_dist = state.distToPuck / 200.0

    if state.player_haspuck == False:
        if state.distToPuck < state.last_dist:
            #rew = 1.0 / (1.0 + scaled_dist)
            rew = 1 - (state.distToPuck / 200.0)**0.5
            #print(state.distToPuck, rew)
        else:
            rew = -0.1
    else:
        rew = 1

    #if state.p1_bodychecks > state.last_p1_bodychecks:
    #    rew = 0.5

    if state.goalie_haspuck:
        rew = -1

    if state.p2_score > state.last_p2_score:
        rew = -1.0

    if state.p2_shots > state.last_p2_shots:
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


    if state.p1_bodychecks > state.last_p1_bodychecks:
        rew = 1.0

    if state.p1_passing > state.last_p1_passing:
        rew = 1.0

    if not state.player_haspuck:
        if state.p1_y > -80:
            rew = -1.0
        if state.puck_y > -80:
            rew = -1.0

    if state.goalie_haspuck:
        rew = -1.0

    if state.p2_score > state.last_p2_score:
        rew = -1.0

    if state.p2_shots > state.last_p2_shots:
        rew = -1.0

    #if state.time < 200:
    #    rew = -1

    return rew

# =====================================================================
# Register Functions
# =====================================================================
def register_functions(name):
    if name == "GetPuck":
        return init_getpuck, rf_getpuck, isdone_getpuck
    elif name == "ScoreGoal":
        return init_scoregoal, rf_scoregoal, isdone_scoregoal
    elif name == "ScoreGoal02":
        return init_scoregoal02, rf_scoregoal02, isdone_scoregoal02
    elif name == "KeepPuck":
        return init_keeppuck, rf_keeppuck, isdone_keeppuck
    elif name == "DefenseZone":
        return init_defensezone, rf_defensezone, isdone_defensezone
    elif name == "General":
        return init_general, rf_general, isdone_general
    else:
        raise Exception("Unsupported Reward Function")

    return none, none, none
