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
class NHL94ObservationEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.last_p1_score = 0
        self.last_p2_score = 0
        self.last_p1_shots = 0
        self.last_p1_bodychecks = 0
        self.last_p1_attackzone = 0
        self.last_p2_attackzone = 0
        self.last_p1_faceoffwon = 0
        self.last_p2_shots = 0
        self.last_p1_pos = (0,0)
        self.last_p2_pos = (0,0)
        self.last_puck_pos = (0,0)
        self.last_time = 0
        self.last_p1_passing = 0
        self.last_dist = 0
        self.counter = 0
        self.lastshot_time = -1
        self.time = 0

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        return self.state, info

    def Distance(self, vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
    
        return math.sqrt(tmp)

    def calc_reward(self, info):
        p1_score = info.get('p1_score')
        p2_score = info.get('p2_score')
        p1_shots = info.get('p1_shots')
        p2_shots = info.get('p2_shots')
        p1_bodychecks = info.get('p1_bodychecks')
        p2_attackzone = info.get('p2_attackzone')
        p1_attackzone = info.get('p1_attackzone')
        p1_faceoffwon = info.get('p1_faceoffwon')
        p1_passing = info.get('p1_passing')
        p1_x = info.get('p1_x')
        p1_y = info.get('p1_y')
        p2_x = info.get('p2_x')
        p2_y = info.get('p2_y')
        g1_x = info.get('g1_x')
        g1_y = info.get('g1_y')
        time = info.get('time')
        puck_x = info.get('puck_x')
        puck_y = info.get('puck_y')
        
        #print(p1_x, p1_y, p2_x, p2_y)

        fullstar_x = info.get('fullstar_x')
        fullstar_y = info.get('fullstar_y')

        #print()
        
        player_haspuck = False
        goalie_haspuck = False

        if(p1_x == fullstar_x and p1_y == fullstar_y):
            player_haspuck = True
        elif(g1_x == fullstar_x and g1_y == fullstar_y):
            goalie_haspuck = True



        isGoodShot = True
        hasPuck = False
        rew = 0

        distToPuck = self.Distance((p1_x, p1_y), (puck_x, puck_y))


        # if player_haspuck == False:
        #     if p1_y > 120 and p1_shots > self.last_p1_shots:
        #         self.lastshot_time = self.time
        #         rew = 1.0
        #     elif distToPuck < self.last_dist:
        #         rew = 0.3
        #     else:
        #         rew = -1

        #if p1_bodychecks > self.last_p1_bodychecks:
        #    rew = 0.5
        # else:
        #     rew = 1.0

        #     if p1_y > 120: rew = 0.0

        #if p1_faceoffwon > self.last_p1_faceoffwon:
        #     rew = 1.0
                
        
        # if self.lastshot_time != -1:
        #     if (self.time - self.lastshot_time > 60):
        #         self.lastshot_time = -1
        #     else:
        #         rew = 1.0
        

        

        #if p1_attackzone > self.last_p1_attackzone and p1_shots > self.last_p1_shots:
        #    rew = 0.2

        if p1_score > self.last_p1_score:
            rew = 1.0

        if p1_shots > self.last_p1_shots:
            rew = 0.1
     

        #if p2_attackzone > self.last_p2_attackzone:
        #   rew = -0.2
            
        #if p2_score > self.last_p2_score:
        #    rew = -1.0

        #if p2_shots > self.last_p2_shots:
        #    rew = -1.0


        self.last_p1_score = p1_score
        self.last_p1_shots = p1_shots
        self.last_p1_bodychecks = p1_bodychecks
        self.last_p2_attackzone = p2_attackzone
        self.last_p1_attackzone = p1_attackzone
        self.last_p1_faceoffwon = p1_faceoffwon
        self.last_p2_shots = p2_shots
        self.last_p2_score = p2_score
        self.last_time = time
        self.last_p1_passing = p1_passing
        self.last_dist = distToPuck

        #last_p2_pos


        #if abs(p1_x - p2_x) > 50 or abs(p1_y - p2_y) > 50:
        #    rew = -1.0
        #elif abs(p1_x - p2_x) > 30 or abs(p1_y - p2_y) > 30:
        #    rew = 0.5
        #else:
        #    rew = 1.0


        # Don't give rewards when clock is not running
        #if time == self.last_time:
        #    rew = 0

        #if abs(g1_x - puck_x) <= 6 and abs(g1_y - puck_y) <= 6:
        #    rew = -1

        return rew


    def step(self, ac):
        ob, rew, terminated, truncated, info = self.env.step(ac)


        

        #rew = 1000

        #print(rew)

        #print(info.get('p2_x'), info.get('p2_y'))
        #print((self.last_p2_pos[0] - info.get('p2_x')), (self.last_p2_pos[1] - info.get('p2_y')))

        #time.sleep(0.01)
        

        #done = True

        rew = self.calc_reward(info)

        p1_score = info.get('p1_score')
        p1_x = info.get('p1_x') / 120
        p1_y = info.get('p1_y') / 300
        p2_x = info.get('p2_x') / 120
        p2_y = info.get('p2_y') / 300
        g2_x = info.get('g2_x') / 120
        g2_y = info.get('g2_y') / 300

        puck_x = info.get('puck_x') / 120
        puck_y = info.get('puck_y') / 300


        p1_velx = (self.last_p1_pos[0] - info.get('p1_x')) / 80
        p1_vely = (self.last_p1_pos[1] - info.get('p2_y')) / 80
        p2_velx = (self.last_p2_pos[0] - info.get('p2_x')) / 80
        p2_vely = (self.last_p2_pos[1] - info.get('p2_y')) / 80
        puck_velx = (self.last_puck_pos[0] - puck_x) / 80
        puck_vely = (self.last_puck_pos[1] - puck_y) / 80
        puck_velx = info.get('puck_vel_x')  / 50
        puck_vely = info.get('puck_vel_y')  / 50

        #print("=======")
        #print((self.last_p2_pos[0], info.get('p2_x')))
        #print((p2_velx))
        #print((self.last_p2_pos[1], info.get('p2_y')))
        #print((p2_vely))


        self.state = (p1_x, p1_y, \
                     p1_velx, p1_vely, \
                     p2_x, p2_y, \
                     p2_velx, p2_vely, \
                     puck_x, puck_y, \
                     puck_velx, puck_vely, \
                     g2_x, g2_y)

        self.counter += 1
        if self.counter == 10:
            self.last_p1_pos = (info.get('p1_x'), info.get('p1_y'))
            self.last_p2_pos = (info.get('p2_x'), info.get('p2_y'))
            self.last_puck_pos = (info.get('puck_x'), info.get('puck_y'))
            self.counter = 0
        #print(self.state)

        ob = self.state
        #print(ob)

        #print(rew)

        self.time += 1


        if puck_y < -10:
            terminated = True


        return ob, rew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)
