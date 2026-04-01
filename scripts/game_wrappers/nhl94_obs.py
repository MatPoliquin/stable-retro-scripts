"""
NHL94 Observation wrapper
"""

import datetime
import os
import random
import copy
from collections import deque
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
        if args.env == 'NHL941on1-Genesis-v0':
            self.num_players_per_team = 1
        elif args.env == 'NHL942on2-Genesis-v0':
            self.num_players_per_team = 2
        elif args.env == 'NHL94-Genesis-v0':
            self.num_players_per_team = 5

        self.NUM_PARAMS = self.init_model(self.num_players_per_team)

        self.game_state = NHL94GameState(self.num_players_per_team)
        self.uses_sequence_obs = self.nn in ('HybridMambaPolicy', 'GRUMlpPolicy')
        self.frame_stack_size = max(1, int(getattr(args, 'seq_len', 16)))
        self.frame_buffer = deque(maxlen=self.frame_stack_size) if self.uses_sequence_obs else None

        low = np.array([-1] * self.NUM_PARAMS, dtype=np.float32)
        high = np.array([1] * self.NUM_PARAMS, dtype=np.float32)

        if self.nn == 'CombinedPolicy':
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(224, 256, 3), dtype=np.uint8),
                'scalar': spaces.Box(low, high, dtype=np.float32)
            })
        elif self.uses_sequence_obs:
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.frame_stack_size, self.NUM_PARAMS),
                dtype=np.float32,
            )
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

        # Per-side action processing state so learner and opponent debounce
        # counters do not corrupt each other during self-play.
        self.action_state = {
            "learner": {"b_pressed": False, "c_pressed": False, "slapshot_frames": 0},
            "opponent": {"b_pressed": False, "c_pressed": False, "slapshot_frames": 0},
        }

        self.SLAPSHOT_HOLD_FRAMES = 60     # Number of frames to hold C for slapshot

        # Self-play state -------------------------------------------------
        # Activated via set_selfplay_role() / set_opponent_model().
        # Team 1 is always the learner in the first implementation.
        self.selfplay_enabled = False
        self.selfplay_role = None          # "offense" or "defense"
        self.opponent_model_path = None
        self.opponent_model = None
        # Exposed so external callers (e.g. curriculum) can adjust this.
        self.control_frames_required = 60
        self.selfplay_outcome = None       # set to +1 / -1 / 0 at episode end

    def _get_scalar_state_array(self):
        return np.asarray(self.state, dtype=np.float32)

    def _reset_frame_buffer(self):
        if not self.uses_sequence_obs:
            return

        self.frame_buffer.clear()
        current_state = self._get_scalar_state_array()
        for _ in range(self.frame_stack_size):
            self.frame_buffer.append(current_state.copy())

    def _get_obs(self, image_obs=None):
        if self.nn == 'CombinedPolicy':
            return {
                'image': image_obs,
                'scalar': self.state
            }
        if self.uses_sequence_obs:
            return np.array(self.frame_buffer, dtype=np.float32, copy=True)
        return self.state

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.state = tuple([0] * self.NUM_PARAMS)
        self._reset_frame_buffer()

        self.game_state = NHL94GameState(self.num_players_per_team)
        self.ram_inited = False
        self.action_state = {
            "learner": {"b_pressed": False, "c_pressed": False, "slapshot_frames": 0},
            "opponent": {"b_pressed": False, "c_pressed": False, "slapshot_frames": 0},
        }
        self.selfplay_outcome = None

        return self._get_obs(state), info

    def _process_filtered_action(self, ac, side: str):
        """Process a 12-button FILTERED action for *side* ('learner' or 'opponent').

        Returns the processed action list and a 6-element gamestate_ac vector.
        The per-side action_state dict is updated in place.
        """
        ac = list(ac)
        st = self.action_state[side]

        # B button debounce
        if st["b_pressed"] and ac[GameConsts.INPUT_B] == 1:
            ac[GameConsts.INPUT_B] = 0
            st["b_pressed"] = False
        elif not st["b_pressed"] and ac[GameConsts.INPUT_B] == 1:
            st["b_pressed"] = True
        else:
            st["b_pressed"] = False

        # C button / slapshot handling
        if ac[GameConsts.INPUT_MODE] == 1:
            if st["slapshot_frames"] == 0:
                st["slapshot_frames"] = 1
                ac[GameConsts.INPUT_C] = 1
            else:
                st["slapshot_frames"] += 1
                ac[GameConsts.INPUT_C] = 1
                if st["slapshot_frames"] >= self.SLAPSHOT_HOLD_FRAMES:
                    st["slapshot_frames"] = 0
                    ac[GameConsts.INPUT_C] = 0
        else:
            if st["c_pressed"] and ac[GameConsts.INPUT_C] == 1:
                ac[GameConsts.INPUT_C] = 0
                st["c_pressed"] = False
            elif not st["c_pressed"] and ac[GameConsts.INPUT_C] == 1:
                st["c_pressed"] = True
            else:
                st["c_pressed"] = False
            st["slapshot_frames"] = 0

        gamestate_ac = [
            ac[GameConsts.INPUT_UP] == 1,
            ac[GameConsts.INPUT_DOWN] == 1,
            ac[GameConsts.INPUT_LEFT] == 1,
            ac[GameConsts.INPUT_RIGHT] == 1,
            ac[GameConsts.INPUT_B] == 1,
            ac[GameConsts.INPUT_C] == 1,
        ]
        return ac, gamestate_ac

    def _process_multidiscrete_action(self, ac, side: str):
        """Process a 3-button MULTI_DISCRETE action for *side*.

        Returns the processed action list and a 6-element gamestate_ac vector.
        The per-side action_state dict is updated in place.
        """
        ac = list(ac)
        st = self.action_state[side]

        # B button debounce (action == 1)
        if ac[2] == 1:
            if st["b_pressed"]:
                ac[2] = 0
                st["b_pressed"] = False
            else:
                st["b_pressed"] = True
        else:
            st["b_pressed"] = False

        # C button / slapshot (action == 2)
        if ac[2] == 2:
            if st["slapshot_frames"] == 0:
                st["slapshot_frames"] = 1
            else:
                st["slapshot_frames"] += 1
                if st["slapshot_frames"] >= self.SLAPSHOT_HOLD_FRAMES:
                    ac[2] = 0
                    st["slapshot_frames"] = 0
        else:
            st["slapshot_frames"] = 0

        gamestate_ac = [
            ac[0] == 1,
            ac[0] == 2,
            ac[1] == 1,
            ac[1] == 2,
            ac[2] == 1,
            ac[2] == 2,
        ]
        return ac, gamestate_ac

    # ------------------------------------------------------------------
    # Self-play interface
    # ------------------------------------------------------------------

    def set_opponent_model(self, path: str) -> None:
        """Load (or swap) the frozen opponent policy from *path*.

        Designed to be called via ``SubprocVecEnv.env_method`` so that every
        worker subprocess loads the model independently.  Passing ``None`` or
        an empty string disables the opponent and falls back to no-op actions.
        """
        if not path:
            self.opponent_model = None
            self.opponent_model_path = None
            return

        zip_path = path if path.endswith(".zip") else f"{path}.zip"
        if not os.path.exists(zip_path):
            self.opponent_model = None
            self.opponent_model_path = None
            return

        try:
            from stable_baselines3 import PPO  # imported lazily to avoid hard dep
            self.opponent_model = PPO.load(zip_path)
            self.opponent_model_path = zip_path
        except Exception:  # pylint: disable=broad-except
            self.opponent_model = None
            self.opponent_model_path = None

    def set_selfplay_role(self, role) -> None:
        """Set the self-play drill role for this env.

        *role* must be ``"offense"``, ``"defense"``, or ``None``.
        Setting a non-None role also enables self-play.
        """
        assert role in ("offense", "defense", None), \
            f"selfplay role must be 'offense', 'defense', or None — got {role!r}"
        self.selfplay_role = role
        self.selfplay_enabled = role is not None

    def compute_opponent_action(self, obs):
        """Return an action array from the frozen opponent model.

        If no opponent model is loaded, a no-op 12-button action is returned.
        The observation is passed as-is (team-1 perspective for Milestone 1).
        """
        if self.opponent_model is None:
            return [0] * 12

        obs_array = np.array(obs, dtype=np.float32)
        action, _ = self.opponent_model.predict(obs_array, deterministic=True)
        return action

    def combine_selfplay_actions(self, learner_action, opponent_action):
        """Build the two-player action payload consumed by the retro env."""
        return np.concatenate(
            [np.array(learner_action), np.array(opponent_action)]
        )

    # ------------------------------------------------------------------

    def step(self, ac):
        gamestate_ac = [0] * 6

        # Process learner action
        if isinstance(ac, (list, np.ndarray)) and len(ac) == 12:
            ac, gamestate_ac = self._process_filtered_action(ac, "learner")
        elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
            ac, gamestate_ac = self._process_multidiscrete_action(ac, "learner")
        else:
            raise ValueError(f"Unsupported action format: {ac}")

        # Reward functions might need to override input
        self.input_overide(ac)

        # Build full two-player action for the retro env
        if self.selfplay_enabled:
            # Compute and process the frozen opponent's action (team 2)
            opp_raw = self.compute_opponent_action(np.array(self.state, dtype=np.float32))
            if isinstance(opp_raw, (list, np.ndarray)) and len(opp_raw) == 12:
                opp_ac, _ = self._process_filtered_action(opp_raw, "opponent")
            elif isinstance(opp_raw, (list, np.ndarray)) and len(opp_raw) == 3:
                opp_ac, _ = self._process_multidiscrete_action(opp_raw, "opponent")
            else:
                opp_ac = [0] * 12
            ac2 = self.combine_selfplay_actions(ac, opp_ac)
        elif self.num_players == 2:
            p2_ac = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ac2 = np.concatenate([ac, np.array(p2_ac)])
        else:
            ac2 = ac

        ob, rew, terminated, truncated, info = self.env.step(ac2)

        if not self.ram_inited:
            self.init_function(self.env, self.env_name)
            self.ram_inited = True

        self.prev_state = copy.deepcopy(self.game_state)

        self.game_state.BeginFrame(info, gamestate_ac)

        # Calculate Reward and check if episode is done
        rew = self.reward_function(self.game_state)
        terminated = self.done_function(self.game_state)

        if terminated:
            self.selfplay_outcome = rew

        self.game_state.EndFrame()

        # ============================
        # SET MODEL INPUT
        # ============================
        self.state = self.set_model_input(self.game_state)
        if self.uses_sequence_obs:
            self.frame_buffer.append(self._get_scalar_state_array().copy())

        return self._get_obs(ob), rew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)
