"""
NHL94 Reward Functions
"""

import random
import numpy as np
import math
from typing import Tuple, Callable
from game_wrappers.nhl94_const import GameConsts
from game_wrappers.nhl94_mi import init_model, init_model_rel, init_model_rel_dist, init_model_rel_dist_buttons, init_model_1p, init_model_2p, \
      set_model_input, set_model_input_1p, set_model_input_2p, set_model_input_rel, set_model_input_rel_dist, set_model_input_rel_dist_buttons, \
      init_model_invariant, set_model_input_invariant

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
    y = (random.random() * (GameConsts.SPAWNABLE_ZONE_HEIGHT + 100)) + 0

    #print(x,y)
    return x, y

def RandomPosDefenseZone():
    x = (random.random() - 0.5) * GameConsts.SPAWNABLE_AREA_WIDTH
    y = (random.random() * GameConsts.SPAWNABLE_ZONE_HEIGHT) - 220

    #print(x,y)
    return x, y

def input_overide(ac):
    if isinstance(ac, (list, np.ndarray)) and len(ac) == 12:
        ac[GameConsts.INPUT_B] = 0
        ac[GameConsts.INPUT_C] = 0
    elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
        ac[2] = 0

def input_overide_no_shoot(ac):
    if isinstance(ac, (list, np.ndarray)) and len(ac) == 12:
        ac[GameConsts.INPUT_C] = 0
    elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
        if ac[2] == 2:
            ac[2] = 0

def input_overide_empty(ac):
    return

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
    elif env_name == 'NHL94-Genesis':
        # Team 1 players
        for i in range(5):
            x, y = RandomPosAttackZone()
            #print(x,y)
            if i == 0:
                env.set_value("p1_x", x)
                env.set_value("p1_y", y)
            else:
                env.set_value(f"p1_{i+1}_x", x)
                env.set_value(f"p1_{i+1}_y", y)

        # Team 2 players
        for i in range(5):
            x, y = RandomPosAttackZone()
            #x, y = RandomPosDefenseZone()
            if i == 0:
                env.set_value("p2_x", x)
                env.set_value("p2_y", y)
            else:
                env.set_value(f"p2_{i+1}_x", x)
                env.set_value(f"p2_{i+1}_y", y)
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
    
    # Scoring rewards (team 1 scores or concedes)
    if t1.stats.score > t1.last_stats.score:
        return 1.0  # Big reward for scoring
    if t2.stats.score > t2.last_stats.score:
        return -1.0  # Penalty for conceding

    reward = 0.0

    # Possession shaping
    if t1.player_haspuck or t1.goalie_haspuck:
        reward += 0.12
    elif t2.player_haspuck or t2.goalie_haspuck:
        reward -= 0.1

    # Encourage advancing the puck into the offensive zone
    zone_span = GameConsts.ATACKZONE_POS_Y - GameConsts.DEFENSEZONE_POS_Y
    if zone_span:
        puck_progress = (state.puck.y - GameConsts.DEFENSEZONE_POS_Y) / zone_span
        puck_progress = float(np.clip(puck_progress, 0.0, 1.0))
        reward += 0.1 * (puck_progress * 2.0 - 1.0)

    # Keep skater moving, especially with possession
    controlled_player = t1.goalie if t1.control == 0 else t1.players[t1.control - 1]
    speed = math.sqrt(controlled_player.vx ** 2 + controlled_player.vy ** 2)
    speed_norm = min(1.0, speed / GameConsts.MAX_VEL_XY)
    if t1.player_haspuck:
        reward += 0.05 * speed_norm
    else:
        reward += 0.02 * speed_norm

    # Reward purposeful passing (limit farming by requiring forward/zone progress)
    passes_delta = max(0, t1.stats.passing - t1.last_stats.passing)
    if passes_delta:
        in_attack_zone = state.puck.y >= GameConsts.ATACKZONE_POS_Y
        forward_velocity = state.puck.vy > 0
        pass_bonus = 0.12 if in_attack_zone or forward_velocity else 0.04
        reward += pass_bonus * min(passes_delta, 2)

    # Reward generating shots, penalize conceded shots
    if t1.stats.shots > t1.last_stats.shots:
        reward += 0.25
    if t2.stats.shots > t2.last_stats.shots:
        reward -= 0.25

    # Encourage time in offensive zone, discourage defending too long
    attackzone_delta = max(0, t1.stats.attackzone - t1.last_stats.attackzone)
    if attackzone_delta:
        reward += min(0.15, 0.01 * attackzone_delta)

    opponent_zone_delta = max(0, t2.stats.attackzone - t2.last_stats.attackzone)
    if opponent_zone_delta:
        reward -= min(0.12, 0.008 * opponent_zone_delta)

    # Small living penalty to push toward decisive plays
    reward -= 0.002

    return reward

# =====================================================================
# ScoreGoal - Cross Crease technic
# =====================================================================
def isdone_scoregoal_cc(state):
    t1 = state.team1
    t2 = state.team2

    # Success
    #if t1.stats.score > t1.last_stats.score:
    #    return True

    # Mild failures (just end episode)
    if t2.player_haspuck or t2.goalie_haspuck:
        return True

    if state.puck.y < 100:
        return True

    # Timeout
    if state.time < 100:
        return True

    return False

def rf_scoregoal_cc(state):
    t1 = state.team1
    t2 = state.team2
    rew = 0.0



    # Failure conditions
    if t2.player_haspuck or t2.goalie_haspuck:
        return -0.5  # Less punitive

    if state.puck.y < 100:
        return -0.5

    # Get controlled player
    controlled_player = t1.players[t1.control-1] if t1.control > 0 else t1.goalie

    # Check crease position
    in_crease = (
        controlled_player.y < GameConsts.CREASE_UPPER_BOUND and
        controlled_player.y > GameConsts.CREASE_LOWER_BOUND
    )

    # Reward for entering crease area
    if in_crease:
        #rew += 0.1

        if abs(controlled_player.x) < GameConsts.CREASE_MAX_X * 3:
            rew += 0.1

        # Extra reward for proper positioning (lateral movement)
        if abs(controlled_player.x) < GameConsts.CREASE_MAX_X:
            rew += 0.4

        # Reward for velocity toward net
        #if controlled_player.vy > 0:  # Moving up toward opponent net
        #    rew += 0.1 * min(controlled_player.vy, 1.0)  # Cap velocity reward
        
        if abs(controlled_player.vx) > 15:  # Moving up toward opponent net
            rew = 0.5

    # Reward for shooting when in good position
    if state.action[5] == 1 and in_crease:  # C button pressed (shoot)
        if abs(controlled_player.x - t2.goalie.x) > GameConsts.CREASE_MIN_GOALIE_PUCK_DIST_X:
            rew += 1.0  # Big reward for shooting from good position
        else:
            rew += 0.2  # Small reward for attempting shot

    # Reward for passing (helps set up cross-crease plays)
    if t1.stats.passing > t1.last_stats.passing:
        rew += 0.5

    # Success condition
    if t1.stats.score > 0:
        return 1.0  # Big reward for scoring

    return rew

# =====================================================================
# ScoreGoal - One Timers
# =====================================================================
def isdone_scoregoal_ot(state):
    t1 = state.team1
    t2 = state.team2

    #if t1.stats.score > t1.last_stats.score:
    #    return True

    if t2.goalie_haspuck:
        return True

    if state.puck.y < 0:
        return True

    if state.time < 100:
        return True

    return False

def rf_scoregoal_ot(state):
    t1 = state.team1
    t2 = state.team2
    rew = 0.0

    #if t1.stats.passing > t1.last_stats.passing:
    #    rew = 0.1

    #if t1.stats.shots > t1.last_stats.shots:
    #    rew = 0.1

    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0
    player = t1.players[controlled_idx]

    speed = math.sqrt(player.vx**2 + player.vy**2)
    #print(speed)
    #if t1.player_haspuck and speed > 20:
    #    rew = 1.0

    if speed > 15:
        rew += 0.1

    if t2.player_haspuck:
        rew = -0.1

    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew = 1.0
    
    if t1.stats.score > 0: #and t1.stats.onetimer > 0:
        rew = 1.0

    if state.puck.y < 0:
        rew = -0.3

    return rew

# =====================================================================
# ScoreGoal
# =====================================================================
def isdone_scoregoal(state):
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

def rf_scoregoal(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    #if t2.player_haspuck or t2.goalie_haspuck:
    #    rew = -0.5

    #if state.puck.y < 100:
    #    rew = -1.0

    #if t1.stats.passing > t1.last_stats.passing:
    #    rew = 0.1

    #if t1.stats.shots > t1.last_stats.shots:
    #    rew = 0.1

    #if t1.stats.score > t1.last_stats.score:
    if t1.stats.onetimer > t1.last_stats.onetimer:
        rew = 0.1

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

    if t1.player_haspuck == False:
        return True
    
    if state.puck.y < 100:
        return True

def rf_keeppuck(state):
    t1 = state.team1
    t2 = state.team2

    rew = 1.0
    if not t1.player_haspuck:
        rew = -1.0

    if state.puck.y < 100:
        rew = -1.0

    # Encourage movement; penalize stagnation
    # Use controlled skater if available, otherwise first skater
    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0
    player = t1.players[controlled_idx]

    speed = math.sqrt(player.vx**2 + player.vy**2)
    speed_norm = min(1.0, speed / GameConsts.MAX_VEL_XY)

    motion_bonus = 0.5 * speed_norm         # scale movement reward
    low_speed_threshold = 0.15               # ~7.5% of max speed
    stagnation_penalty = 0.5 if speed_norm < low_speed_threshold else 0.0

    if t1.player_haspuck and state.puck.y >= 100:
        #rew += motion_bonus
        rew -= stagnation_penalty

    if state.puck.x >= 125 or state.puck.x <= -125:
        rew = -0.5


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
def init_defensezone(env, env_name):
    if env_name == 'NHL941on1-Genesis':
        x, y = RandomPosDefenseZone()
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = RandomPosDefenseZone()
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)

    elif env_name == 'NHL942on2-Genesis':
        x, y = RandomPosDefenseZone()
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = RandomPosDefenseZone()
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        x, y = RandomPosDefenseZone()
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        x, y = RandomPosDefenseZone()
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)
    elif env_name == 'NHL94-Genesis':
        # Team 1 players
        for i in range(5):
            x, y = RandomPosDefenseZone()
            #print(x,y)
            if i == 0:
                env.set_value("p1_x", x)
                env.set_value("p1_y", y)
            else:
                env.set_value(f"p1_{i+1}_x", x)
                env.set_value(f"p1_{i+1}_y", y)

        # Team 2 players
        for i in range(5):
            x, y = RandomPosDefenseZone()
            #x, y = RandomPosDefenseZone()
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

    if state.puck.y >= 100:
        return True

    if t2.stats.score > t2.last_stats.score:
        return True

    if state.time < 100:
        return True

def rf_defensezone(state):
    t1 = state.team1
    t2 = state.team2

    tracker = getattr(state, "t1", None)

    # Track short-lived possessions so dumping the puck is penalized
    carry_state = getattr(state, "_defensezone_carry", None)
    if carry_state is None:
        carry_state = {
            "last_player_haspuck": t1.player_haspuck,
            "possession_frames": 1 if t1.player_haspuck else 0,
            "carry_start_y": state.puck.y if t1.player_haspuck else None,
            "last_carry_y": state.puck.y if t1.player_haspuck else None,
        }
        setattr(state, "_defensezone_carry", carry_state)

    def _closest_team_distance(team, puck):
        members = []
        goalie = getattr(team, "goalie", None)
        if goalie is not None:
            members.append(goalie)
        members.extend(getattr(team, "players", []))
        distances = [
            math.hypot(player.x - puck.x, player.y - puck.y)
            for player in members
            if player is not None
        ]
        return min(distances) if distances else None

    dist_to_puck = getattr(tracker, "distToPuck", None)
    if dist_to_puck is None:
        dist_to_puck = _closest_team_distance(t1, state.puck)

    last_dist_to_puck = getattr(tracker, "last_distToPuck", None)
    if last_dist_to_puck is None:
        last_dist_to_puck = dist_to_puck

    reward = 0.0

    if not t1.player_haspuck:
        if dist_to_puck is not None and last_dist_to_puck is not None:
            if dist_to_puck < last_dist_to_puck:
                reward += 1 - (dist_to_puck / 200.0) ** 0.5
            else:
                reward -= 0.1
        else:
            reward -= 0.05

        if carry_state.get("last_player_haspuck", False):
            possession_frames = carry_state.get("possession_frames", 0)
            last_carry_y = carry_state.get("last_carry_y")
            if last_carry_y is None:
                last_carry_y = state.puck.y
            carry_start_y = carry_state.get("carry_start_y")
            if carry_start_y is None:
                carry_start_y = last_carry_y
            forward_gain = last_carry_y - carry_start_y
            launched_forward = state.puck.vy > 12 and state.puck.y > last_carry_y
            still_inside_zone = last_carry_y < GameConsts.DEFENSEZONE_POS_Y
            if possession_frames < 20 and forward_gain < 25 and launched_forward and still_inside_zone:
                reward -= 1.0

        carry_state["possession_frames"] = 0
        carry_state["carry_start_y"] = None
        carry_state["last_carry_y"] = None
    else:
        if not carry_state.get("last_player_haspuck", False):
            carry_state["carry_start_y"] = state.puck.y
            carry_state["possession_frames"] = 0

        carry_state["possession_frames"] = carry_state.get("possession_frames", 0) + 1
        carry_state["last_carry_y"] = state.puck.y

        progress = 0.0
        if carry_state.get("carry_start_y") is not None:
            progress = state.puck.y - carry_state["carry_start_y"]
        if progress > 0:
            reward += min(0.3, 0.02 * (progress / 5.0))

        reward += 0.1

        if state.action[5] and state.puck.y < GameConsts.DEFENSEZONE_POS_Y + 15:
            reward -= 0.4

        if state.puck.y >= -100:
            start_y = carry_state.get("carry_start_y")
            exit_requires_carry = (
                progress >= 20
                or carry_state["possession_frames"] >= 20
                or (start_y is not None and start_y >= GameConsts.DEFENSEZONE_POS_Y - 15)
            )
            if exit_requires_carry:
                reward += 1.0
            else:
                reward -= 0.3
        
    if t1.stats.bodychecks > t1.last_stats.bodychecks:
        reward += 1.0

    if t1.stats.passing > t1.last_stats.passing:
        reward += 1.0

    if not t1.player_haspuck:
        first_player = t1.players[0] if t1.players else None
        if first_player is not None and first_player.y > -80:
            reward -= 1.0
        if state.puck.y > -80:
            reward -= 1.0

    if t1.goalie_haspuck:
        reward -= 1.0

    if t2.stats.score > t2.last_stats.score:
        reward -= 1.0

    if t2.stats.shots > t2.last_stats.shots:
        reward -= 0.1

    carry_state["last_player_haspuck"] = t1.player_haspuck

    return reward

# =====================================================================
# Passing
# =====================================================================
def isdone_passing(state):
    t1 = state.team1
    t2 = state.team2

    if state.puck.y < 100:
        return True

    if state.time < 100:
        return True

    return False

def rf_passing(state):
    t1 = state.team1
    t2 = state.team2

    rew = 0.0

    #if state.puck.y < 100:
    #    rew = -1.0

    #if t1.stats.score > t1.last_stats.score:
    #    rew = -1.0

    if t1.stats.passing > t1.last_stats.passing:
        rew = 1.0

    #rew -= 0.001

    return rew

# =====================================================================
# Self Play
# =====================================================================
def isdone_selfplay(state):
    t1 = state.team1
    t2 = state.team2

    if t1.stats.score > t1.last_stats.score:
        return True

    if state.time < 100:
        return True

    return False

def init_selfplay(env, env_name):
    """Same random (or attack-zone) init as normal."""
    init_general(env, env_name)      # or init_attackzone(...)

def rf_selfplay(state):
    """
    Zero-sum reward wrapper around any existing reward.
    The wrapper decides which side is 'active' and flips the sign.
    """
    # base reward computed from Team-1 perspective
    base = rf_general(state)         # or rf_scoregoal_cc, etc.
    # wrapper will negate if training Team-2
    return base

# =====================================================================
# Register Functions
# =====================================================================
_reward_function_map = {
    "GetPuck": (init_getpuck, rf_getpuck, isdone_getpuck, init_model, set_model_input, input_overide),
    "ScoreGoalCC": (init_attackzone, rf_scoregoal_cc, isdone_scoregoal_cc, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons, input_overide_empty),
    "ScoreGoalOT": (init_attackzone, rf_scoregoal_ot, isdone_scoregoal_ot, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons, input_overide_empty),
    "ScoreGoal": (init_attackzone, rf_scoregoal, isdone_scoregoal, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons, input_overide_empty),
    "KeepPuck": (init_attackzone, rf_keeppuck, isdone_keeppuck, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons, input_overide_empty),
    "DefenseZone": (init_defensezone, rf_defensezone, isdone_defensezone, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons, input_overide_empty),
    "Passing": (init_attackzone, rf_passing, isdone_passing, init_model_rel_dist, set_model_input_rel_dist, input_overide_no_shoot),
    "General": (init_general, rf_general, isdone_general, init_model_rel_dist_buttons, set_model_input_rel_dist_buttons, input_overide_empty),
    "SelfPlay": (init_selfplay, rf_selfplay, isdone_selfplay, init_model_invariant, set_model_input_invariant, input_overide_empty),
}

def register_functions(name: str) -> Tuple[Callable, Callable, Callable]:
    if name not in _reward_function_map:
        raise ValueError(f"Unsupported Reward Function: {name}")
    return _reward_function_map[name]
