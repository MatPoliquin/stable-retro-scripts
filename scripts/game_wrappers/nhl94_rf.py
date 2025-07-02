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

def init_model_1p():
    return 16

def set_model_input_1p(game_state):
    t1 = game_state.team1
    t2 = game_state.team2

    return (t1.nz_players[0].x, t1.nz_players[0].y, \
                        t1.nz_players[0].vx, t1.nz_players[0].vy, \
                        t2.nz_players[0].x, t2.nz_players[0].y, \
                        t2.nz_players[0].vx, t2.nz_players[0].vy, \
                        game_state.nz_puck.x, game_state.nz_puck.y, \
                        game_state.nz_puck.vx, game_state.nz_puck.vy, \
                        t2.nz_goalie.x, t2.nz_goalie.y, \
                        t1.nz_player_haspuck, t2.nz_goalie_haspuck)

def init_model_2p():
    return 24

def set_model_input_2p(game_state):
    t1 = game_state.team1
    t2 = game_state.team2

    p1_x, p1_y = t1.nz_players[0].x, t1.nz_players[0].y
    p1_vel_x, p1_vel_y = t1.nz_players[0].vx, t1.nz_players[0].vy
    p1_2_x, p1_2_y = t1.nz_players[1].x, t1.nz_players[1].y
    p1_2_vel_x, p1_2_vel_y = t1.nz_players[1].vx, t1.nz_players[1].vy

    # First two slots is for pos/vel of player beeing controled (empty or full star)
    # So swap them if necessary
    if t1.control == 2:
        p1_x, p1_2_x = p1_2_x, p1_x
        p1_y, p1_2_y = p1_2_y, p1_y
        p1_vel_x, p1_2_vel_x = p1_2_vel_x, p1_vel_x
        p1_vel_y, p1_2_vel_y = p1_2_vel_y, p1_vel_y

    return (p1_x, p1_y, \
            p1_vel_x, p1_vel_y, \
            p1_2_x, p1_2_y, \
            p1_2_vel_x, p1_2_vel_y, \
            t2.nz_players[0].x, t2.nz_players[0].y, \
            t2.nz_players[0].vx, t2.nz_players[0].vy, \
            t2.nz_players[1].x, t2.nz_players[1].y, \
            t2.nz_players[1].vx, t2.nz_players[1].vy, \
            game_state.nz_puck.x, game_state.nz_puck.y, \
            game_state.nz_puck.vx, game_state.nz_puck.vy, \
            t2.nz_goalie.x, t2.nz_goalie.y, \
            t1.nz_player_haspuck, t2.nz_goalie_haspuck)



# =====================================================================
# Common RF functions
# =====================================================================
def init_attackzone(env, env_name):

    if env_name == 'NHL941on1-Genesis':
        x, y = RandomPosAttackZone()
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = RandomPosAttackZone()
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)

    elif env_name == 'NHL942on2-Genesis':
        x, y = RandomPosAttackZone()
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = RandomPosAttackZone()
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        x, y = RandomPosAttackZone()
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        x, y = RandomPosAttackZone()
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)
    else:
        raise ValueError(f"Invalid environment name, got '{env_name}'")


# =====================================================================
# General - One model for both offense and defense
# =====================================================================
def init_general(env, env_name):
    if env_name == 'NHL941on1-Genesis':
        x, y = RandomPos()
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = RandomPos()
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
    elif env_name == 'NHL942on2-Genesis':
        x, y = RandomPos()
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = RandomPos()
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        x, y = RandomPos()
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        x, y = RandomPos()
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)

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
    
    # Big rewards for scoring/conceding
    if t1.stats.score > t1.last_stats.score:
        rew += 1.0
    if t2.stats.score > t2.last_stats.score:
        rew -= 1.0
    
    # Determine if we should focus on offense or defense
    offensive_mode = (t1.player_haspuck or 
                     (not t2.player_haspuck and state.puck.y > -50))
    
    if offensive_mode:
        # Offensive rewards
        if t1.stats.shots > t1.last_stats.shots:
            rew += 0.2
        if t1.stats.passing > t1.last_stats.passing:
            rew += 0.1
        
        # Reward for moving puck toward opponent goal
        if t1.player_haspuck:
            if state.puck.vy < 0:
                rew += 0.01 * state.puck.vy
                
        # Cross-crease opportunity reward
        if (t1.player_haspuck and 
            t1.players[t1.control-1].y < GameConsts.CREASE_UPPER_BOUND and 
            t1.players[t1.control-1].y > GameConsts.CREASE_LOWER_BOUND and
            abs(state.puck.x - t2.goalie.x) > GameConsts.CREASE_MIN_GOALIE_PUCK_DIST_X):
            rew += 0.3
    else:
        # Defensive rewards
        if t1.stats.bodychecks > t1.last_stats.bodychecks:
            rew += 0.2
            
        # Reward for moving toward puck when we don't have it
        if not t1.player_haspuck and t1.distToPuck < t1.last_distToPuck:
            rew += 0.1 * (1 - (t1.distToPuck / 200.0)**0.5)
            
        # Penalty for opponent shots
        if t2.stats.shots > t2.last_stats.shots:
            rew -= 0.1
            
        # Reward for clearing puck from our zone
        #if state.puck.y < state.last_puck_y and state.puck.y < -50:
        #    rew += 0.05
    
    return rew

# =====================================================================
# ScoreGoal - Cross Crease technic
# =====================================================================
def isdone_scoregoal_cc(state):
    t1 = state.team1
    t2 = state.team2

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
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    # reward scoring opportunities
    t = t1.control - 1
    assert t == 0 if state.numPlayers == 1 else True
 
    if t1.player_haspuck and t1.players[t].y < GameConsts.CREASE_UPPER_BOUND and t1.players[t].y  > GameConsts.CREASE_LOWER_BOUND:
        rew = 0.1
        if t1.players[t].vx >= GameConsts.CREASE_MIN_VEL or t1.players[t].vx <= -GameConsts.CREASE_MIN_VEL:
            rew = 0.3
            if state.puck.x > -GameConsts.CREASE_MAX_X and state.puck.x < GameConsts.CREASE_MAX_X:
                if abs(state.puck.x - t2.goalie.x) > GameConsts.CREASE_MIN_GOALIE_PUCK_DIST_X:
                    rew = 1.0
                else:
                    rew = 0.5

    return rew

# =====================================================================
# ScoreGoal - One Timers
# =====================================================================
def isdone_scoregoal_ot(state):
    t1 = state.team1
    t2 = state.team2

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.puck.y < 100:
        return True

    if state.time < 100:
        return True

    return False

def rf_scoregoal_ot(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    if t2.player_haspuck or t2.goalie_haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    #if t1.stats.passing > t1.last_stats.passing:
    #    rew = 0.1

    # reward one timers
    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew = 1.0

    return rew

# =====================================================================
# ScoreGoal
# =====================================================================
def isdone_scoregoal(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score: #or self.game_state.p1_shots > self.game_state.last_p1_shots:
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

    if t2.player_haspuck or t2.goalie_haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    #if t1.stats.passing > t1.last_stats.passing:
    #    rew = 0.1

    #if t1.stats.shots > t1.last_stats.shots:
    #    rew = 0.1

    if t1.stats.score > t1.last_stats.score:
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

    if t1.player_haspuck == False:
        if t1.distToPuck < t1.last_distToPuck:
            #rew = 1.0 / (1.0 + scaled_dist)
            rew = 1 - (t1.distToPuck / 200.0)**0.5
            #print(state.distToPuck, rew)
        else:
            rew = -0.1
    else:
        rew = 1


    if t1.stats.bodychecks > t1.last_stats.bodychecks:
        rew = 1.0

    if t1.stats.passing > t1.last_stats.passing:
        rew = 1.0

    if not t1.player_haspuck:
        if t1.players[0].y > -80:
            rew = -1.0
        if state.puck.y > -80:
            rew = -1.0

    if t1.goalie_haspuck:
        rew = -1.0

    if t2.stats.score > t2.last_stats.score:
        rew = -1.0

    if t2.stats.shots > t2.last_stats.shots:
        rew = -1.0

    #if state.time < 200:
    #    rew = -1

    return rew

# =====================================================================
# Passing
# =====================================================================
def init_model_2p_passing():
    return 19

def set_model_input_2p_passing(game_state):
    t1 = game_state.team1
    t2 = game_state.team2

    p1_x, p1_y = t1.nz_players[0].x, t1.nz_players[0].y
    p1_vel_x, p1_vel_y = t1.nz_players[0].vx, t1.nz_players[0].vy
    p1_2_x, p1_2_y = t1.nz_players[1].x, t1.nz_players[1].y
    p1_2_vel_x, p1_2_vel_y = t1.nz_players[1].vx, t1.nz_players[1].vy

    # First two slots is for pos/vel of player beeing controled (empty or full star)
    # So swap them if necessary
    if t1.control == 2:
        p1_x, p1_2_x = p1_2_x, p1_x
        p1_y, p1_2_y = p1_2_y, p1_y
        p1_vel_x, p1_2_vel_x = p1_2_vel_x, p1_vel_x
        p1_vel_y, p1_2_vel_y = p1_2_vel_y, p1_vel_y

    return (p1_x, p1_y, \
            p1_vel_x, p1_vel_y, \
            p1_2_x, p1_2_y, \
            p1_2_vel_x, p1_2_vel_y, \
            t2.nz_players[0].x, t2.nz_players[0].y, \
            t2.nz_players[0].vx, t2.nz_players[0].vy, \
            t2.nz_players[1].x, t2.nz_players[1].y, \
            t2.nz_players[1].vx, t2.nz_players[1].vy, \
            t1.nz_player_haspuck, \
            t2.nz_player_haspuck,
            t2.nz_goalie_haspuck)

def init_passing(env):
    x, y = RandomPosAttackZone()
    env.set_value("p2_x", x)
    env.set_value("p2_y", y)

    x, y = RandomPosAttackZone()
    env.set_value("p2_2_x", x)
    env.set_value("p2_2_y", y)

    x, y = RandomPosAttackZone()
    env.set_value("p1_x", x)
    env.set_value("p1_y", y)

    x, y = RandomPosAttackZone()
    env.set_value("p1_2_x", x)
    env.set_value("p1_2_y", y)

def isdone_passing(state):
    t1 = state.team1
    t2 = state.team2

    if state.puck.y < 100:
        return True

    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.time < 100:
        return True

    return False

def rf_passing(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    if t2.player_haspuck or t2.goalie_haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    if t1.stats.passing > t1.last_stats.passing:
        if t1.players[0].y > GameConsts.CREASE_UPPER_BOUND and t1.players[1].y > GameConsts.CREASE_UPPER_BOUND:
            if (t1.players[0].x < -GameConsts.CREASE_MAX_X and t1.players[1].x > GameConsts.CREASE_MAX_X) or \
            (t1.players[1].x < -GameConsts.CREASE_MAX_X and t1.players[0].x > GameConsts.CREASE_MAX_X):
                rew = 1.0
        rew = 0.2

    return rew

# =====================================================================
# Register Functions
# =====================================================================
_reward_function_map = {
    "GetPuck_1P": (init_getpuck, rf_getpuck, isdone_getpuck, init_model_1p, set_model_input_1p),
    "GetPuck_2P": (init_getpuck, rf_getpuck, isdone_getpuck, init_model_2p, set_model_input_2p),
    "ScoreGoalCC": (init_attackzone, rf_scoregoal_cc, isdone_scoregoal_cc, init_model_1p, set_model_input_1p),
    "ScoreGoalOT": (init_attackzone, rf_scoregoal_ot, isdone_scoregoal_ot, init_model_2p, set_model_input_2p),
    "ScoreGoal": (init_attackzone, rf_scoregoal, isdone_scoregoal, init_model_2p, set_model_input_2p),
    "KeepPuck_1P": (init_keeppuck, rf_keeppuck, isdone_keeppuck, init_model_1p, set_model_input_1p),
    "DefenseZone_1P": (init_defensezone, rf_defensezone, isdone_defensezone, init_model_1p, set_model_input_1p),
    "DefenseZone_2P": (init_defensezone, rf_defensezone, isdone_defensezone, init_model_2p, set_model_input_2p),
    "Passing_2P": (init_passing, rf_passing, isdone_passing, init_model_2p_passing, set_model_input_2p_passing),
    "General": (init_general, rf_general, isdone_general, init_model_2p, set_model_input_2p),
}

def register_functions(name: str) -> Tuple[Callable, Callable, Callable]:
    if name not in _reward_function_map:
        raise ValueError(f"Unsupported Reward Function: {name}")
    return _reward_function_map[name]
