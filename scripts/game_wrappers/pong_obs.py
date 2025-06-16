"""
Pong Observation wrapper
"""
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
import numpy as np

class PongObservationEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        self.NUM_PARAMS = 6
        self.MAX_XY = 210
        self.MAX_VEL_XY = 2

        self.nn = args.nn

        low = np.array([-1] * self.NUM_PARAMS, dtype=np.float32)
        high = np.array([1] * self.NUM_PARAMS, dtype=np.float32)

        if self.nn == 'CombinedPolicy':
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8),
                'scalar': spaces.Box(low, high, dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.last_ball_x = 0
        self.last_ball_y = 0
        self.last_p1_score = 0
        self.last_p2_score = 0

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = tuple([0] * self.NUM_PARAMS)

        if self.nn == 'CombinedPolicy':
             return {
                 'image': state,
                 'scalar': self.state
             }, info
        else:
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

        rew = self.calc_reward(info)

        ball_x = info.get('ball_x')
        ball_y = info.get('ball_y')
        p1_y = info.get('p1_pos')
        p2_y = info.get('p2_pos')

        ball_velx = (self.last_ball_x - ball_x)
        ball_vely = (self.last_ball_y - ball_y)

        self.state = (p1_y / self.MAX_XY, p2_y / self.MAX_XY, \
                     ball_x / self.MAX_XY, ball_y / self.MAX_XY, \
                     ball_velx / self.MAX_VEL_XY, ball_vely / self.MAX_VEL_XY)

        self.last_ball_x = ball_x
        self.last_ball_y = ball_y

        if self.nn == 'CombinedPolicy':
            return {
                 'image': ob,
                 'scalar': self.state
            }, rew, terminated, truncated, info
        else:
            return self.state, rew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)
