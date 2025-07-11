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


class NHL94Observation2PEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        self.nn = args.nn
        self.env_name = args.env

        self.rf_name = rf_name
        self.init_function, self.reward_function, self.done_function, self.init_model, self.set_model_input, self.input_overide = register_functions(self.rf_name)

        self.num_players_per_team = 0
        if args.env == 'NHL941on1-Genesis':
            self.num_players_per_team = 1
        elif args.env == 'NHL942on2-Genesis':
            self.num_players_per_team = 2
        elif args.env == 'NHL94-Genesis':
            self.num_players_per_team = 5

        self.NUM_PARAMS = self.init_model(self.num_players_per_team)

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

        self.slapshot_frames_held = 0      # 0 means not in slapshot mode
        self.SLAPSHOT_HOLD_FRAMES = 60     # Number of frames to hold C for slapshot

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

        # Handle different action space types
        if isinstance(ac, (list, np.ndarray)) and len(ac) == 12:
            # FILTERED action space (12-button array)
            # B button handling
            if self.b_button_pressed and ac[GameConsts.INPUT_B] == 1:
                ac[GameConsts.INPUT_B] = 0
                self.b_button_pressed = False
            elif not self.b_button_pressed and ac[GameConsts.INPUT_B] == 1:
                self.b_button_pressed = True
            else:
                self.b_button_pressed = False

            # C button handling (slapshot)
            if ac[GameConsts.INPUT_MODE] == 1:
                if self.slapshot_frames_held == 0:
                    self.slapshot_frames_held = 1
                    ac[GameConsts.INPUT_C] = 1
                else:
                    self.slapshot_frames_held += 1
                    ac[GameConsts.INPUT_C] = 1
                    if self.slapshot_frames_held >= self.SLAPSHOT_HOLD_FRAMES:
                        self.slapshot_frames_held = 0
                        ac[GameConsts.INPUT_C] = 0
            else:
                if self.c_button_pressed and ac[GameConsts.INPUT_C] == 1:
                    ac[GameConsts.INPUT_C] = 0
                    self.c_button_pressed = False
                elif not self.c_button_pressed and ac[GameConsts.INPUT_C] == 1:
                    self.c_button_pressed = True
                else:
                    self.c_button_pressed = False
                self.slapshot_frames_held = 0
        elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
            # MULTI_DISCRETE action space (3-button array)
            # Process MULTI_DISCRETE actions for NHL94 format:
            # ac = [vertical, horizontal, action] where:
            #   0 = no input
            #   1 = first option (e.g., "UP")
            #   2 = second option (e.g., "DOWN")

            # Make a copy to avoid modifying original
            processed_ac = list(ac).copy()

            # B button handling (action button 1)
            if processed_ac[2] == 1:  # B pressed
                if self.b_button_pressed:
                    processed_ac[2] = 0  # Release B if already pressed
                    self.b_button_pressed = False
                else:
                    self.b_button_pressed = True
            else:
                self.b_button_pressed = False

            # C button handling (action button 2 - slapshot)
            if processed_ac[2] == 2:  # C pressed
                if self.slapshot_frames_held == 0:
                    self.slapshot_frames_held = 1
                else:
                    self.slapshot_frames_held += 1
                    if self.slapshot_frames_held >= self.SLAPSHOT_HOLD_FRAMES:
                        processed_ac[2] = 0  # Release C after hold duration
                        self.slapshot_frames_held = 0
            else:
                self.slapshot_frames_held = 0

            ac = processed_ac


        else:
            # Handle other cases or raise error
            raise ValueError(f"Unsupported action format: {ac}")




        # Reward functions might need to override input
        #print(ac)
        self.input_overide(ac)

        ac2 = ac
        if self.num_players == 2:
            ac2 = np.concatenate([ac, np.array(p2_ac)])

        ob, rew, terminated, truncated, info = self.env.step(ac2)

        if not self.ram_inited:
            self.init_function(self.env, self.env_name)
            self.ram_inited = True

        self.prev_state = copy.deepcopy(self.game_state)

        self.game_state.BeginFrame(info)

        # Calculate Reward and check if episode is done
        rew = self.reward_function(self.game_state)
        terminated = self.done_function(self.game_state)

        self.game_state.EndFrame()

        # ============================
        # SET MODEL INPUT
        # ============================
        self.state = self.set_model_input(self.game_state)
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
