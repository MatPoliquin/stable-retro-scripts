"""
NHL94 Observation wrapper
"""

import os, datetime
import retro
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np
from os import environ
import math
import time
import random
import copy
from datetime import datetime

from game_wrappers.nhl94_const import GameConsts

from game_wrappers.nhl94_rf import rf_general, rf_getpuck, rf_keeppuck, rf_scoregoal, rf_defensezone, isdone_getpuck, isdone_scoregoal, isdone_keeppuck, isdone_defensezone, init_getpuck, init_scoregoal, init_keeppuck, init_defensezone
from game_wrappers.nhl94_ai import NHL94AISystem

from game_wrappers.nhl94_gamestate import NHL94GameState

class NHL94Observation2PEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reward_function = None

        #self.action_space = 12 * [0]

        self.prev_state = None

        self.target_xy = [-1, -1]
        random.seed(datetime.now().timestamp())

        self.num_players = num_players
        if num_players == 2:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons)

        self.game_state = NHL94GameState()

        self.ai_sys = NHL94AISystem(args, env, None)

        self.ram_inited = False

        self.rf_name = rf_name
        self.reward_function = None
        self.done_function = None
        self.init_function = None

        if self.rf_name == "GetPuck":
            self.init_function = init_getpuck
            self.reward_function = rf_getpuck
            self.done_function = isdone_getpuck
        elif self.rf_name == "ScoreGoal":
            self.init_function = init_scoregoal
            self.reward_function = rf_scoregoal
            self.done_function = isdone_scoregoal
        elif self.rf_name == "KeepPuck":
            self.init_function = init_keeppuck
            self.reward_function = rf_keeppuck
            self.done_function = isdone_keeppuck
        elif self.rf_name == "DefenseZone":
            self.init_function = init_defensezone
            self.reward_function = rf_defensezone
            self.done_function = isdone_defensezone
        else:
            print('error')
            #rew = self.calc_reward_general(info)
            #if self.game_state.time < 10:
            #    terminated = True

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        self.game_state = NHL94GameState()
        self.ram_inited = False

        return self.state, info

    def step(self, ac):
        p2_ac = [0,0,0,0,0,0,0,0,0,0,0,0]
        p1_zero = [0,0,0,0,0,0,0,0,0,0,0,0]

        if self.prev_state != None:
            self.prev_state.Flip()
            p2_ac = self.ai_sys.Think_GotoRandomTarget(self.prev_state)
        
        #ac2 = [0,0,0,0,0,0,0,0,0,0,0,0] + p2_ac
        ac2 = np.concatenate([ac, np.array(p2_ac)])


        ob, rew, terminated, truncated, info = self.env.step(ac2)

        if not self.ram_inited:
            self.init_function(self.env)
            self.ram_inited = True



        self.prev_state = copy.deepcopy(self.game_state)

        self.game_state.BeginFrame(info)
        
        # Calculate Reward and check if episode is done
        rew = self.reward_function(self.game_state)
        terminated = self.done_function(self.game_state)
       
        self.game_state.EndFrame()

        self.state = (self.game_state.normalized_p1_x, self.game_state.normalized_p1_y, \
                     self.game_state.normalized_p1_velx, self.game_state.normalized_p1_vely, \
                     self.game_state.normalized_p2_x, self.game_state.normalized_p2_y, \
                     self.game_state.normalized_p2_velx, self.game_state.normalized_p2_vely, \
                     self.game_state.normalized_puck_x, self.game_state.normalized_puck_y, \
                     self.game_state.normalized_puck_velx, self.game_state.normalized_puck_vely, \
                     self.game_state.normalized_g2_x, self.game_state.normalized_g2_y, \
                     self.game_state.normalized_player_haspuck, self.game_state.normalized_goalie_haspuck)
        
        ob = self.state
        
        return ob, rew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)
