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

from game_wrappers.nhl94_rf import rf_general, rf_getpuck, rf_keeppuck, rf_scoregoal, isdone_getpuck, isdone_scoregoal, isdone_keeppuck
from game_wrappers.nhl94_ai import NHL94AISystem

from game_wrappers.nhl94_gamestate import NHL94GameState

#https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
class NHL94Discretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(NHL94Discretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['A'], ['DOWN', 'LEFT'], ['DOWN', 'RIGHT'], ['UP', 'LEFT'], ['UP', 'RIGHT']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


# WARNING: NON FUNCTIONAL CODE - WIP
class NHL94Observation2PEnv(gym.Wrapper):
    def __init__(self, env, args, num_players):
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

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        self.game_state = NHL94GameState()
        self.ram_inited = False

        return self.state, info

    def set_reward_function(self, rf):
        self.reward_function = rf

    def RandomPos(self):
        x = (random.random() - 0.5) * 240
        y = (random.random() - 0.5) * 460

        #print(x,y)
        return x, y

    def init_ramvalues(self):
        #x, y = self.RandomPos()
        #self.env.set_value("rpuck_x", x)
        #self.env.set_value("rpuck_y", y)

        x, y = self.RandomPos()
        self.env.set_value("rp2_x", x)
        self.env.set_value("rp2_y", y)

        x, y = self.RandomPos()
        self.env.set_value("p1_x", x)
        self.env.set_value("rp1_y", y)
        


    def step(self, ac):
        p2_ac = [0,0,0,0,0,0,0,0,0,0,0,0]
        p1_zero = [0,0,0,0,0,0,0,0,0,0,0,0]

        if self.prev_state != None:
            self.prev_state.Flip()
            p2_ac = self.ai_sys.Think_testAI(self.prev_state)
            #self.ai_sys.
        
        #ac2 = [0,0,0,0,0,0,0,0,0,0,0,0] + p2_ac
        ac2 = np.concatenate([ac, np.array(p2_ac)])

        #print("Hello")
        

        ob, rew, terminated, truncated, info = self.env.step(ac2)

        if not self.ram_inited:
            self.init_ramvalues()
            self.ram_inited = True


        #self.env.set_value("rp2_x", 0)
        #self.env.set_value("rp2_y", 0)

        self.prev_state = copy.deepcopy(self.game_state)
        

        self.game_state.BeginFrame(info)
        
        # Calculate Reward and check if episode is done
        if self.reward_function == "GetPuck":
            rew = rf_getpuck(self.game_state)
            terminated = isdone_getpuck(self.game_state)
        elif self.reward_function == "ScoreGoal":
            rew = rf_scoregoal(self.game_state)
            terminated = isdone_scoregoal(self.game_state)
        elif self.reward_function == "KeepPuck":
            rew = rf_keeppuck(self.game_state)
            terminated = isdone_keeppuck(self.game_state)
        else:
            #print('error')
            rew = self.calc_reward_general(info)
            if self.game_state.time < 10:
                terminated = True


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
