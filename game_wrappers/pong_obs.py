"""
NHL94 Observation wrapper
"""

import gymnasium as gym
from gymnasium import spaces
from gym.spaces import Dict, Box
import numpy as np

class PongObservationEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        self.nn = args.nn

        print(self.observation_space)

        if self.nn == 'CombinedPolicy':
            low = np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32)
            high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8),
                'scalar': spaces.Box(low, high, dtype=np.float32)
            })
        else:
            low = np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32)
            high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.last_ball_x = 0
        self.last_ball_y = 0
        self.last_p1_score = 0
        self.last_p2_score = 0

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = (0, 0, 0, 0, 0, 0)


        if self.nn == 'CombinedPolicy':
             return {
                 'image': state,
                 'scalar': self.state
             }, info
             #print('COMBINED')
        else:
            #print(self.nn)
            return self.state, info

    def calc_reward(self, info):
        p1_score = info.get('score2')
        p2_score = info.get('score1')

        rew = 0

        if(p1_score > self.last_p1_score):
            rew=1.0
        if(p2_score > self.last_p2_score):
            rew=-1.0

        self.last_p1_score = p1_score
        self.last_p2_score = p2_score

        return rew

    def step(self, ac):
        ob, rew, terminated, truncated, info = self.env.step(ac)
        #print("=======================================================")
        #print(ob)

        rew = self.calc_reward(info)

        ball_x = info.get('ball_x')
        ball_y = info.get('ball_y')
        p1_y = info.get('p1_pos')
        p2_y = info.get('p2_pos')

        ball_velx = (self.last_ball_x - ball_x)
        ball_vely = (self.last_ball_y - ball_y)

        #print(ball_velx, ball_vely)


        self.state = (p1_y / 210, p2_y / 210, \
                     ball_x / 210, ball_y / 210, \
                     ball_velx / 2, ball_vely / 2)

        #print(self.state)

        self.last_ball_x = ball_x
        self.last_ball_y = ball_y

        #print(ob)
        #print(rew)

        if self.nn == 'CombinedPolicy':
            return {
                 'image': ob,
                 'scalar': self.state
            }, rew, terminated, truncated, info
            #ob = {
            #     'image': ob,
            #     'scalar': self.state
            #print('COMBINED')
        else:
            #ob = self.state
            return self.state, rew, terminated, truncated, info
            #print(self.nn)

    def seed(self, s):
        self.rng.seed(s)
