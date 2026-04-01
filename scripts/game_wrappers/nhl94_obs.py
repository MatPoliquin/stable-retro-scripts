"""
NHL94 Observation wrapper
"""

import random
from collections import deque
from datetime import datetime
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game_wrappers.nhl94_const import GameConsts
from game_wrappers.nhl94_rf import register_functions
from game_wrappers.nhl94_ai import NHL94AISystem
from game_wrappers.nhl94_gamestate import NHL94GameState


def _make_action_side_state():
    """Return a fresh per-side action-processing state dict."""
    return {"b_pressed": False, "c_pressed": False, "slapshot_frames": 0}


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

        self.target_xy = [-1, -1]
        random.seed(datetime.now().timestamp())

        self.num_players = num_players
        if num_players == 2:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons)

        self.ai_sys = NHL94AISystem(args, env, None)

        self.ram_inited = False

        # Per-side action-processing state (learner = team 1, opponent = team 2)
        self.action_state = {
            "learner": _make_action_side_state(),
            "opponent": _make_action_side_state(),
        }

        self.SLAPSHOT_HOLD_FRAMES = 60     # Number of frames to hold C for slapshot

        # Self-play fields
        self.opponent_model_path: str = ""
        self.opponent_model = None

    # ------------------------------------------------------------------
    # Self-play public API
    # ------------------------------------------------------------------

    def set_opponent_model(self, path: str) -> None:
        """Load or swap the frozen opponent model from ``path``.

        Pass an empty string to disable the frozen opponent.
        """
        from stable_baselines3 import PPO  # pylint: disable=import-outside-toplevel

        self.opponent_model_path = path
        if path:
            self.opponent_model = PPO.load(path)
        else:
            self.opponent_model = None

    def compute_opponent_action(self, obs: np.ndarray) -> np.ndarray:
        """Query the frozen opponent policy for an action.

        The opponent receives the current (team-1-perspective) observation.
        Returns a zero-filled action array when no opponent model is loaded.
        """
        if self.opponent_model is None:
            if hasattr(self.action_space, 'n'):
                return np.zeros(1, dtype=np.int64)
            return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        action, _ = self.opponent_model.predict(obs, deterministic=True)
        return action

    def combine_selfplay_actions(self, learner_ac, opponent_ac) -> np.ndarray:
        """Combine learner and opponent actions into a two-player action array."""
        return np.concatenate([np.array(learner_ac), np.array(opponent_ac)])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def selfplay_enabled(self) -> bool:
        """True when a frozen opponent model has been loaded."""
        return self.opponent_model is not None

    def _process_action(self, ac, side_state):
        """Apply button debounce and slapshot handling in-place for one side.

        Returns ``(updated_ac, gamestate_ac)`` where ``gamestate_ac`` is the
        6-element boolean action array consumed by the game-state tracker.
        """
        gamestate_ac = [0] * 6

        if isinstance(ac, (list, np.ndarray)) and len(ac) == 12:
            # B button debounce
            if side_state["b_pressed"] and ac[GameConsts.INPUT_B] == 1:
                ac[GameConsts.INPUT_B] = 0
                side_state["b_pressed"] = False
            elif not side_state["b_pressed"] and ac[GameConsts.INPUT_B] == 1:
                side_state["b_pressed"] = True
            else:
                side_state["b_pressed"] = False

            # C button / slapshot
            if ac[GameConsts.INPUT_MODE] == 1:
                if side_state["slapshot_frames"] == 0:
                    side_state["slapshot_frames"] = 1
                    ac[GameConsts.INPUT_C] = 1
                else:
                    side_state["slapshot_frames"] += 1
                    ac[GameConsts.INPUT_C] = 1
                    if side_state["slapshot_frames"] >= self.SLAPSHOT_HOLD_FRAMES:
                        side_state["slapshot_frames"] = 0
                        ac[GameConsts.INPUT_C] = 0
            else:
                if side_state["c_pressed"] and ac[GameConsts.INPUT_C] == 1:
                    ac[GameConsts.INPUT_C] = 0
                    side_state["c_pressed"] = False
                elif not side_state["c_pressed"] and ac[GameConsts.INPUT_C] == 1:
                    side_state["c_pressed"] = True
                else:
                    side_state["c_pressed"] = False
                side_state["slapshot_frames"] = 0

            gamestate_ac[0] = ac[GameConsts.INPUT_UP] == 1
            gamestate_ac[1] = ac[GameConsts.INPUT_DOWN] == 1
            gamestate_ac[2] = ac[GameConsts.INPUT_LEFT] == 1
            gamestate_ac[3] = ac[GameConsts.INPUT_RIGHT] == 1
            gamestate_ac[4] = ac[GameConsts.INPUT_B] == 1
            gamestate_ac[5] = ac[GameConsts.INPUT_C] == 1

        elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
            processed_ac = list(ac).copy()

            # B button debounce
            if processed_ac[2] == 1:
                if side_state["b_pressed"]:
                    processed_ac[2] = 0
                    side_state["b_pressed"] = False
                else:
                    side_state["b_pressed"] = True
            else:
                side_state["b_pressed"] = False

            # C slapshot
            if processed_ac[2] == 2:
                if side_state["slapshot_frames"] == 0:
                    side_state["slapshot_frames"] = 1
                else:
                    side_state["slapshot_frames"] += 1
                    if side_state["slapshot_frames"] >= self.SLAPSHOT_HOLD_FRAMES:
                        processed_ac[2] = 0
                        side_state["slapshot_frames"] = 0
            else:
                side_state["slapshot_frames"] = 0

            ac = processed_ac

            gamestate_ac[0] = ac[0] == 1
            gamestate_ac[1] = ac[0] == 2
            gamestate_ac[2] = ac[1] == 1
            gamestate_ac[3] = ac[1] == 2
            gamestate_ac[4] = ac[2] == 1
            gamestate_ac[5] = ac[2] == 2

        else:
            raise ValueError(f"Unsupported action format: {ac}")

        return ac, gamestate_ac

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
            "learner": _make_action_side_state(),
            "opponent": _make_action_side_state(),
        }

        return self._get_obs(state), info

    def step(self, ac):
        p2_ac = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Process learner action with side-aware state
        ac = list(ac) if isinstance(ac, np.ndarray) else ac
        ac, gamestate_ac = self._process_action(ac, self.action_state["learner"])

        # Reward functions might need to override input
        self.input_overide(ac)

        if self.num_players == 2:
            if self.selfplay_enabled:
                # Query frozen opponent and process its action independently
                learner_obs = np.array(self.state, dtype=np.float32)
                opp_raw = self.compute_opponent_action(learner_obs)
                opp_raw = list(opp_raw) if isinstance(opp_raw, np.ndarray) else list(opp_raw)
                # Ensure the opponent action has the same length as the learner action
                if len(opp_raw) != len(ac):
                    opp_raw = (opp_raw + [0] * len(ac))[:len(ac)]
                opp_ac, _ = self._process_action(opp_raw, self.action_state["opponent"])
                p2_ac = opp_ac
            ac2 = np.concatenate([np.array(ac), np.array(p2_ac)])
        else:
            ac2 = ac

        ob, rew, terminated, truncated, info = self.env.step(ac2)

        if not self.ram_inited:
            self.init_function(self.env, self.env_name)
            self.ram_inited = True

        self.game_state.BeginFrame(info, gamestate_ac)

        # Calculate Reward and check if episode is done
        rew = self.reward_function(self.game_state)
        terminated = self.done_function(self.game_state)

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
