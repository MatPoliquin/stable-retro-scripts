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
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
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
        self.last_dist = -1
        self.last_dist_az = -1
        self.counter = 0
        self.lastshot_time = -1
        self.time = 0
        self.last_havepuck_time = -1

        self.reward_function = None

        #self.action_space = 12 * [0]

        self.prev_info = None

        self.target_xy = [-1, -1]
        random.seed(datetime.now().timestamp())

        self.action_space = gym.spaces.MultiBinary(self.num_buttons)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        return self.state, info

    def set_reward_function(self, rf):
        self.reward_function = rf


    def Distance(self, vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
    
        return math.sqrt(tmp)

    def calc_reward_general(self, info):
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
        
        player_haspuck = False
        goalie_haspuck = False

        if(p1_x == fullstar_x and p1_y == fullstar_y):
            player_haspuck = True
        elif(g1_x == fullstar_x and g1_y == fullstar_y):
            goalie_haspuck = True



        isGoodShot = True
        rew = 0

        distToPuck = self.Distance((p1_x, p1_y), (puck_x, puck_y))
        distToAttackZone = 120 - p1_y 

        if player_haspuck == False:
            if p1_y > 120 and p1_shots > self.last_p1_shots:
                self.lastshot_time = self.time
                rew = 1.0
            elif self.last_dist != -1:
                if distToPuck < self.last_dist:
                    rew = 0.3
                else:
                    rew = -1
        else:
            if p1_y < 120 and self.last_dist_az != -1:
                if distToAttackZone < self.last_dist_az:
                    rew = 0.3
                else:
                    rew = -1


        if p1_bodychecks > self.last_p1_bodychecks:
            rew = 0.5

        #     if p1_y > 120: rew = 0.0

        #if p1_faceoffwon > self.last_p1_faceoffwon:
        #     rew = 0.5
                
        
        if self.lastshot_time != -1:
             if (self.time - self.lastshot_time > 60):
                 self.lastshot_time = -1
             else:
                 rew = 1.0
        
        if goalie_haspuck:
            rew = -1

        if p1_passing > self.last_p1_passing:
            rew = 0.5
        

        #if p1_attackzone > self.last_p1_attackzone and p1_shots > self.last_p1_shots:
        #    rew = 0.2

        if p1_score > self.last_p1_score:
            rew = 1.0

        #if p1_shots > self.last_p1_shots:
        #    rew = 0.1
     

        #if p2_attackzone > self.last_p2_attackzone:
        #   rew = -0.2
            
        if p2_score > self.last_p2_score:
            rew = -1.0

        if p2_shots > self.last_p2_shots:
            rew = -1.0


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
        self.last_dist_az = distToAttackZone

        #last_p2_pos


        #if abs(p1_x - p2_x) > 50 or abs(p1_y - p2_y) > 50:
        #    rew = -1.0
        #elif abs(p1_x - p2_x) > 30 or abs(p1_y - p2_y) > 30:
        #    rew = 0.5
        #else:
        #    rew = 1.0


        # Don't give rewards when clock is not running
        if puck_x == 0 and puck_y == 0:
            rew = 0

        #if abs(g1_x - puck_x) <= 6 and abs(g1_y - puck_y) <= 6:
        #    rew = -1

        return rew
    
    def calc_reward_scoregoal(self, info):
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

        fullstar_x = info.get('fullstar_x')
        fullstar_y = info.get('fullstar_y')
        
        player_haspuck = False
        goalie_haspuck = False

        isGoodShot = True
        rew = 0

        distToPuck = self.Distance((p1_x, p1_y), (puck_x, puck_y))
        distToAttackZone = 120 - p1_y 

        if p1_y < 120 and self.last_dist_az != -1:
            if distToAttackZone < self.last_dist_az:
                rew = 0.1
            else:
                rew = -1

        if p1_score > self.last_p1_score:
            rew = 1.0

        if p1_y < 120 and p1_shots > self.last_p1_shots:
            rew = 0.3

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
        self.last_dist_az = distToAttackZone

        return rew

    def calc_reward_keeppuck(self, info):
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

        fullstar_x = info.get('fullstar_x')
        fullstar_y = info.get('fullstar_y')
        

        player_haspuck = False
        goalie_haspuck = False

        if(p1_x == fullstar_x and p1_y == fullstar_y):
            player_haspuck = True
        elif(g1_x == fullstar_x and g1_y == fullstar_y):
            goalie_haspuck = True

        rew = 1.0
        if not player_haspuck:
            rew = -1.0

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
        #self.last_dist = distToPuck
        #self.last_dist_az = distToAttackZone

        return rew
    
    def calc_reward_getpuck(self, info):
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
        
        player_haspuck = False
        goalie_haspuck = False

        if(p1_x == fullstar_x and p1_y == fullstar_y):
            player_haspuck = True
        elif(g1_x == fullstar_x and g1_y == fullstar_y):
            goalie_haspuck = True

        rew = 0

        distToPuck = self.Distance((p1_x, p1_y), (puck_x, puck_y))

        scaled_dist = distToPuck / 200.0

        if player_haspuck == False:
            if distToPuck < self.last_dist:
                rew = 1.0 / (1.0 + scaled_dist)
                #print(rew)
            else:
                rew = -1.0
        else:
            rew = 1

        if p1_bodychecks > self.last_p1_bodychecks:
            rew = 0.5

        if goalie_haspuck:
            rew = -1

        if p2_score > self.last_p2_score:
            rew = -1.0

        if p2_shots > self.last_p2_shots:
            rew = -1.0

        if time < 100:
            rew = -1


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

        return rew
    
    def DistToPos(self, vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
    
        return math.sqrt(tmp)

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
        #print(ac)
        
        p2_ac = [0,0,0,0,0,0,0,0,0,0,0,0]

        # up = random.random()
        # right = random.random()

        # if up >= 0.5:
        #     p2_ac[GameConsts.INPUT_UP] = 1
        # else:
        #     p2_ac[GameConsts.INPUT_DOWN] = 1

        # if right >= 0.5:
        #     p2_ac[GameConsts.INPUT_RIGHT] = 1
        # else:
        #     p2_ac[GameConsts.INPUT_LEFT] = 1

        p1_zero = [0,0,0,0,0,0,0,0,0,0,0,0]
        #p2_ac[GameConsts.INPUT_UP] = 1
        #p1_zero[GameConsts.INPUT_LEFT] = 1

        think = random.random()

        if self.prev_info != None: #and think > 0.2:
            p2_ac = self.Think_testAI(self.prev_info)
        
        ac2 = [0,0,0,0,0,0,0,0,0,0,0,0] + p2_ac
        
        ac2 = np.concatenate([ac, np.array(p2_ac)])
        #print(ac2)
        #print(np.array(p2_ac))
        #ac2 = [0,0,0,0,0,0,0,0,0,0,0,0] + [0,0,0,0,1,0,0,0,0,0,0,0]
        #ac2 = p1_zero + p1_zero
        #print(ac2)
        ob, rew, terminated, truncated, info = self.env.step(ac2)

        self.prev_info = info

        time = info.get('time')
        p1_shots = info.get('p1_shots')

        #print(time)

        p1_score = info.get('p1_score')
        p1_x = info.get('p1_x') / 120
        p1_y = info.get('p1_y') / 300
        p2_x = info.get('p2_x') / 120
        p2_y = info.get('p2_y') / 300
        g2_x = info.get('g2_x') / 120
        g2_y = info.get('g2_y') / 300

        puck_x = info.get('puck_x') / 130
        puck_y = info.get('puck_y') / 270

        g1_x = info.get('g1_x')
        g1_y = info.get('g1_y')
        fullstar_x = info.get('fullstar_x')
        fullstar_y = info.get('fullstar_y')


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

        player_haspuck = 0.0
        goalie_haspuck = 0.0

        distToPuck = self.Distance((p1_x, p1_y), (0, 0))

        if(info.get('p1_x') == fullstar_x and info.get('p1_y') == fullstar_y):
            player_haspuck = 1.0
            self.last_havepuck_time = time
        if(g1_x == fullstar_x and g1_y == fullstar_y):
            goalie_haspuck = 1.0


        self.state = (p1_x, p1_y, \
                     p1_velx, p1_vely, \
                     p2_x, p2_y, \
                     p2_velx, p2_vely, \
                     puck_x, puck_y, \
                     puck_velx, puck_vely, \
                     g2_x, g2_y, \
                     player_haspuck, goalie_haspuck)

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


        # Calculate Reward and check if episode is done
        if self.reward_function == "GetPuck":
            #print('GetPuck')
            rew = self.calc_reward_getpuck(info)
            #print(rew)
            if player_haspuck > 0.0:
                #print('TERMINATED: GOT PUCK: (%d,%d) (%d,%d)' % (info.get('p1_x'), info.get('p1_y'), fullstar_x, fullstar_y))
                terminated = True
            if time < 100:
                #print('TERMINATED: TIME')
                terminated = True
        elif self.reward_function == "ScoreGoal":
            rew = self.calc_reward_scoregoal(info)
            if p1_score > self.last_p1_score or p1_shots > self.last_p1_shots:
                if self.last_havepuck_time != -1 and (time - self.last_havepuck_time > 30):
                    terminated = True
        elif self.reward_function == "KeepPuck":
            rew = self.calc_reward_keeppuck(info)
            if player_haspuck == 0.0:
                terminated = True
        else:
            #print('error')
            rew = self.calc_reward_general(info)
            if time < 10:
                terminated = True

        
        return ob, rew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)
