"""
Fighter game Observation wrapper
Supports:
MortalKombatII-Genesis
"""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
import numpy as np

NUM_PARAMS = 2
MIN_X = 284
MAX_X = 1147

class FighterObservationEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        self.nn = args.nn

        low = np.array([-1] * NUM_PARAMS, dtype=np.float32)
        high = np.array([1] * NUM_PARAMS, dtype=np.float32)

        if self.nn == 'CombinedPolicy':
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8),
                'scalar': spaces.Box(low, high, dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.last_p1_health = 0
        self.last_p2_health = 0

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = tuple([0] * NUM_PARAMS)

        if self.nn == 'CombinedPolicy':
             return {
                 'image': state,
                 'scalar': self.state
             }, info
        else:
            return self.state, info

    def calc_reward(self, info):
        p1_health = info.get('health')
        p2_health = info.get('enemy_health')

        rew = 0

        if(p1_health < self.last_p1_health):
            rew = -1.0
        if(p2_health < self.last_p2_health):
            rew = 1.0

        self.last_p1_health = p1_health
        self.last_p2_health = p2_health

        return rew

    def step(self, ac):
        ob, rew, terminated, truncated, info = self.env.step(ac)

        if self.nn == 'CombinedPolicy':
            rew = self.calc_reward(info)

            max_width = (MAX_X - MIN_X)
            p1_pos_x = (info.get('x_position') - MIN_X) / max_width
            #p1_pos_y = info.get('y_position')
            p2_pos_x = (info.get('enemy_x_position') - MIN_X) / max_width
            #p2_pos_y = info.get('enemy_y_position')

            self.state = (p1_pos_x, p2_pos_x)
            return {
                 'image': ob,
                 'scalar': self.state
            }, rew, terminated, truncated, info
        else:
            return ob, rew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)
