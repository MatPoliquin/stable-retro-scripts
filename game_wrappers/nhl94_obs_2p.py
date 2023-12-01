"""
NHL94 Observation wrapper
"""

import os, datetime
import argparse
import retro
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np
from os import environ
import math
import time
import random
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
    def __init__(self, env, num_players):
        gym.Wrapper.__init__(self, env)

        low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reward_function = None

        #self.action_space = 12 * [0]

        self.prev_info = None

        self.target_xy = [-1, -1]
        random.seed(datetime.now().timestamp())

        self.num_players = num_players
        if num_players == 2:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons)

        self.game_state = NHL94GameState()

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        self.game_state = NHL94GameState()

        return self.state, info

    def set_reward_function(self, rf):
        self.reward_function = rf

    def GotoTarget(self, p1_actions, target_vec):
        if target_vec[0] > 0:
            p1_actions[GameConsts.INPUT_LEFT] = 1
        else:
            p1_actions[GameConsts.INPUT_RIGHT] = 1

        if target_vec[1] > 0:
            p1_actions[GameConsts.INPUT_DOWN] = 1
        else:
            p1_actions[GameConsts.INPUT_UP] = 1

    def SelectRandomTarget(self):
        x = (random.random() - 0.5) * 240
        y = (random.random() - 0.5) * 460

        return [x,y]
    
    def DistToPos(self, vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
    
        return math.sqrt(tmp)
    
    def Think_testAI(self, info):
        p1_actions = [0] * GameConsts.INPUT_MAX

        p2_x = info.get('p2_x')
        p2_y = info.get('p2_y')
        g2_x = info.get('g2_x')
        g2_y = info.get('g2_y')
        puck_x = info.get('puck_x')
        puck_y = info.get('puck_y')
        fullstar_x = info.get('p2_fullstar_x')
        fullstar_y = info.get('p2_fullstar_y')

        player_haspuck = False
        goalie_haspuck = False

        if(p2_x == fullstar_x and p2_y == fullstar_y):
            player_haspuck = True
        elif(g2_x == fullstar_x and g2_y == fullstar_y):
            goalie_haspuck = True

        pp_vec = [p2_x - puck_x, p2_y - puck_y]
        tmp = (p2_x - puck_x)**2 + (p2_y - puck_y)**2
        pp_dist = math.sqrt(tmp)

        #if(goalie_haspuck): print('GOALIE HAS PUCK')

        goto_target_test = False

        if player_haspuck:
            if not goto_target_test:
                dist = self.DistToPos([p2_x, p2_y], [GameConsts.SHOOT_POS_X, GameConsts.SHOOT_POS_Y])

                if dist < 60:
                    p1_actions[GameConsts.INPUT_C] = 1
                else:
                    self.GotoTarget(p1_actions, [p2_x - GameConsts.SHOOT_POS_X, p2_y - GameConsts.SHOOT_POS_Y])
                    #print('GOTO SHOOT POSITION')
            else:


                if self.target_xy[0] != -1:
                    dist = self.DistToPos([p2_x, p2_y], self.target_xy)
                else:
                    dist = 0.0

                if dist < 20.0:
                    self.target_xy = self.SelectRandomTarget()
                    #print(self.target_xy)
                    random.seed(datetime.now().timestamp())
                    r = random.random()
                    #print(r)
                    if r < 0.2:
                        p1_actions[GameConsts.INPUT_B] = 1

                # TODO check if target is near opposing player
                self.GotoTarget(p1_actions, [p2_x - self.target_xy[0], p2_y - self.target_xy[1]])
        elif goalie_haspuck:
            p1_actions[GameConsts.INPUT_B] = 1
            #print('GOALIE PASS')
        else:
            self.GotoTarget(p1_actions, pp_vec)
            #print('FIND PUCK')

        return p1_actions

    def step(self, ac):
        p2_ac = [0,0,0,0,0,0,0,0,0,0,0,0]
        p1_zero = [0,0,0,0,0,0,0,0,0,0,0,0]

        if self.prev_info != None:
            p2_ac = self.Think_testAI(self.prev_info)
        
        #ac2 = [0,0,0,0,0,0,0,0,0,0,0,0] + p2_ac
        ac2 = np.concatenate([ac, np.array(p2_ac)])

        ob, rew, terminated, truncated, info = self.env.step(ac2)

        self.prev_info = info
        

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
