"""
NHL94 Observation wrapper
"""

import datetime
import random
import copy
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game_wrappers.nhl94_const import GameConsts
from game_wrappers.nhl94_rf import register_functions
from game_wrappers.nhl94_ai import NHL94AISystem
from game_wrappers.nhl94_gamestate import NHL94GameState

NUM_PARAMS_1ON1 = 16
NUM_PARAMS_2ON2 = 24

class NHL94Observation2PEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        self.nn = args.nn

        self.NUM_PARAMS = 0
        self.num_players_per_team = 0
        if args.env == 'NHL941on1-Genesis':
            self.num_players_per_team = 1
            self.NUM_PARAMS = NUM_PARAMS_1ON1
        else:
            self.num_players_per_team = 2
            self.NUM_PARAMS = NUM_PARAMS_2ON2

        self.game_state = NHL94GameState(self.num_players_per_team)

        low = np.array([-1] * self.NUM_PARAMS, dtype=np.float32)
        high = np.array([1] * self.NUM_PARAMS, dtype=np.float32)

        if self.nn == 'CombinedPolicy':
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(224, 256, 3), dtype=np.uint8),
                'scalar': spaces.Box(low, high, dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reward_function = None

        #self.action_space = 12 * [0]

        self.prev_state = None

        self.target_xy = [-1, -1]
        random.seed(datetime.now().timestamp())

        self.num_players = num_players
        if num_players == 2:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons)

        self.ai_sys = NHL94AISystem(args, env, None)

        self.ram_inited = False
        self.b_button_pressed = False
        self.c_button_pressed = False

        self.rf_name = rf_name
        self.reward_function = None
        self.done_function = None
        self.init_function = None

        self.init_function, self.reward_function, self.done_function = register_functions(self.rf_name)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = tuple([0] * self.NUM_PARAMS)

        self.game_state = NHL94GameState(self.num_players_per_team)
        self.ram_inited = False
        self.b_button_pressed = False
        self.c_button_pressed = False

        #return self.state, info
        if self.nn == 'CombinedPolicy':
             #print(state.shape)
             return {
                 'image': state,
                 'scalar': self.state
             }, info
        else:
            return self.state, info

    def step(self, ac):
        p2_ac = [0,0,0,0,0,0,0,0,0,0,0,0]
        p1_zero = [0,0,0,0,0,0,0,0,0,0,0,0]

        if self.prev_state != None and self.num_players == 2:
            self.prev_state.Flip()

        #ac2 = [0,0,0,0,0,0,0,0,0,0,0,0] + p2_ac
        if self.b_button_pressed and ac[GameConsts.INPUT_B] == 1:
            ac[GameConsts.INPUT_B] = 0
            self.b_button_pressed = False
        elif not self.b_button_pressed and ac[GameConsts.INPUT_B] == 1:
            self.b_button_pressed = True
        else:
            self.b_button_pressed = False

        # Hack to allow for slapshots
        if ac[GameConsts.INPUT_MODE] != 1:
            if self.c_button_pressed and ac[GameConsts.INPUT_C] == 1:
                ac[GameConsts.INPUT_C] = 0
                self.c_button_pressed = False
            elif not self.c_button_pressed and ac[GameConsts.INPUT_C] == 1:
                self.c_button_pressed = True
            else:
                self.c_button_pressed = False

        ac2 = ac
        if self.num_players == 2:
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

        if self.num_players_per_team == 1:
            self.state = (self.game_state.normalized_p1_x, self.game_state.normalized_p1_y, \
                        self.game_state.normalized_p1_velx, self.game_state.normalized_p1_vely, \
                        self.game_state.normalized_p2_x, self.game_state.normalized_p2_y, \
                        self.game_state.normalized_p2_velx, self.game_state.normalized_p2_vely, \
                        self.game_state.normalized_puck_x, self.game_state.normalized_puck_y, \
                        self.game_state.normalized_puck_velx, self.game_state.normalized_puck_vely, \
                        self.game_state.normalized_g2_x, self.game_state.normalized_g2_y, \
                        self.game_state.normalized_player_haspuck, self.game_state.normalized_goalie_haspuck)

        elif self.num_players_per_team == 2:
            p1_x, p1_y = self.game_state.normalized_p1_x, self.game_state.normalized_p1_y
            p1_vel_x, p1_vel_y = self.game_state.normalized_p1_velx, self.game_state.normalized_p1_vely
            p1_2_x, p1_2_y = self.game_state.normalized_p1_2_x, self.game_state.normalized_p1_2_y
            p1_2_vel_x, p1_2_vel_y = self.game_state.normalized_p1_2_velx, self.game_state.normalized_p1_2_vely

            # First two slots is for pos/vel of player beeing controled (empty or full star)
            # So swap them if necessary
            if self.game_state.p1_control == 2:
                p1_x, p1_2_x = p1_2_x, p1_x
                p1_y, p1_2_y = p1_2_y, p1_y
                p1_vel_x, p1_2_vel_x = p1_2_vel_x, p1_vel_x
                p1_vel_y, p1_2_vel_y = p1_2_vel_y, p1_vel_y


            self.state = (p1_x, p1_y, \
                        p1_vel_x, p1_vel_y, \
                        p1_2_x, p1_2_y, \
                        p1_2_vel_x, p1_2_vel_y, \
                        self.game_state.normalized_p2_x, self.game_state.normalized_p2_y, \
                        self.game_state.normalized_p2_velx, self.game_state.normalized_p2_vely, \
                        self.game_state.normalized_p2_2_x, self.game_state.normalized_p2_2_y, \
                        self.game_state.normalized_p2_2_velx, self.game_state.normalized_p2_2_vely, \
                        self.game_state.normalized_puck_x, self.game_state.normalized_puck_y, \
                        self.game_state.normalized_puck_velx, self.game_state.normalized_puck_vely, \
                        self.game_state.normalized_g2_x, self.game_state.normalized_g2_y, \
                        self.game_state.normalized_player_haspuck, self.game_state.normalized_goalie_haspuck)

        #ob = self.state

        #return ob, rew, terminated, truncated, info
        if self.nn == 'CombinedPolicy':
            return {
                 'image': ob,
                 'scalar': self.state
            }, rew, terminated, truncated, info
        else:
            return self.state, rew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)
