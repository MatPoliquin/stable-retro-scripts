"""
NHL94 Reward Functions
"""

import random
from game_wrappers.nhl94_gamestate import NHL94GameState

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

def rf_general(state):
    
    player_haspuck = False
    goalie_haspuck = False

    isGoodShot = True
    rew = 0


    # distToAttackZone = 120 - state.p1_y 

    # if player_haspuck == False:
    #     if p1_y > 120 and p1_shots > self.last_p1_shots:
    #         self.lastshot_time = self.time
    #         rew = 1.0
    #     elif self.last_dist != -1:
    #         if distToPuck < self.last_dist:
    #             rew = 0.3
    #         else:
    #             rew = -1
    # else:
    #     if p1_y < 120 and self.last_dist_az != -1:
    #         if distToAttackZone < self.last_dist_az:
    #             rew = 0.3
    #         else:
    #             rew = -1


    # if p1_bodychecks > self.last_p1_bodychecks:
    #     rew = 0.5

    # #     if p1_y > 120: rew = 0.0

    # #if p1_faceoffwon > self.last_p1_faceoffwon:
    # #     rew = 0.5
            
    
    # if self.lastshot_time != -1:
    #     if (self.time - self.lastshot_time > 60):
    #         self.lastshot_time = -1
    #     else:
    #         rew = 1.0
    
    # if goalie_haspuck:
    #     rew = -1

    # if p1_passing > self.last_p1_passing:
    #     rew = 0.5
    

    # #if p1_attackzone > self.last_p1_attackzone and p1_shots > self.last_p1_shots:
    # #    rew = 0.2

    # if p1_score > self.last_p1_score:
    #     rew = 1.0

    # #if p1_shots > self.last_p1_shots:
    # #    rew = 0.1


    # #if p2_attackzone > self.last_p2_attackzone:
    # #   rew = -0.2
        
    # if p2_score > self.last_p2_score:
    #     rew = -1.0

    # if p2_shots > self.last_p2_shots:
    #     rew = -1.0


    # self.last_p1_score = p1_score
    # self.last_p1_shots = p1_shots
    # self.last_p1_bodychecks = p1_bodychecks
    # self.last_p2_attackzone = p2_attackzone
    # self.last_p1_attackzone = p1_attackzone
    # self.last_p1_faceoffwon = p1_faceoffwon
    # self.last_p2_shots = p2_shots
    # self.last_p2_score = p2_score
    # self.last_time = time
    # self.last_p1_passing = p1_passing
    # self.last_dist = distToPuck
    # self.last_dist_az = distToAttackZone

    # #last_p2_pos


    # #if abs(p1_x - p2_x) > 50 or abs(p1_y - p2_y) > 50:
    # #    rew = -1.0
    # #elif abs(p1_x - p2_x) > 30 or abs(p1_y - p2_y) > 30:
    # #    rew = 0.5
    # #else:
    # #    rew = 1.0


    # # Don't give rewards when clock is not running
    # if puck_x == 0 and puck_y == 0:
    #     rew = 0

    # #if abs(g1_x - puck_x) <= 6 and abs(g1_y - puck_y) <= 6:
    # #    rew = -1

    return rew

def init_scoregoal(env):
    #x, y = self.RandomPos()
    #self.env.set_value("rpuck_x", x)
    #self.env.set_value("rpuck_y", y)

    x, y = RandomPosAttackZone()
    env.set_value("p2_x", x)
    env.set_value("p2_y", y)

    x, y = RandomPosAttackZone()
    env.set_value("p1_x", x)
    env.set_value("p1_y", y)

def isdone_scoregoal(state):
    if state.p1_score > state.last_p1_score: #or self.game_state.p1_shots > self.game_state.last_p1_shots:
        return True

    if state.p2_haspuck:
        return True
    
    if state.puck_y < 100:
        return True

    return False
    
def rf_scoregoal(state):
    
    rew = 0.0

    if state.p2_haspuck:
        rew = -1.0
    
    if state.puck_y < 100:
        rew = -1.0
    
    if state.p1_score > state.last_p1_score: 
        rew = 1.0

    #TODO reward good shots
    #or self.game_state.p1_shots > self.game_state.last_p1_shots:

    return rew

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
    if state.player_haspuck == True:
        #print('TERMINATED: GOT PUCK: (%d,%d) (%d,%d)' % (info.get('p1_x'), info.get('p1_y'), fullstar_x, fullstar_y))
        return True
    elif state.time < 200:
        return True
    
def rf_getpuck(state):
    rew = 0

    scaled_dist = state.distToPuck / 200.0

    if state.player_haspuck == False:
        if state.distToPuck < state.last_dist:
            rew = 1.0 / (1.0 + scaled_dist)
            #print(rew)
        else:
            rew = -1.0
    else:
        rew = 1

    if state.p1_bodychecks > state.last_p1_bodychecks:
        rew = 0.5

    if state.goalie_haspuck:
        rew = -1

    if state.p2_score > state.last_p2_score:
        rew = -1.0

    if state.p2_shots > state.last_p2_shots:
        rew = -1.0

    if state.time < 200:
        rew = -1

    return rew