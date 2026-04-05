"""MKII observation wrapper."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from game_wrappers.mk2.mk2_actions import MK2MacroActionTranslator

NUM_PARAMS = 16
MIN_X = 284
MAX_X = 1147
MAX_HEALTH = 255.0
MAX_ROUNDS = 2.0
Y_POS_SCALE = 512.0
VEL_SCALE = 128.0
DAMAGE_DEALT_REWARD = 1.0
DAMAGE_TAKEN_PENALTY = 0.5
BUTTON_HOLD_FRAMES = 5
BLOCK_HOLD_FRAMES = 4


def _clip_unit(value):
    return float(np.clip(value, -1.0, 1.0))


def _clip_zero_one(value):
    return float(np.clip(value, 0.0, 1.0))


class MK2ObservationEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        self.nn = args.nn
        self.action_type = getattr(args, 'action_type', 'FILTERED').upper()
        self.num_players = num_players
        self.button_names = [str(name) for name in getattr(env, 'buttons', [])]
        self.button_combos = self._coerce_button_combos(getattr(env, 'button_combos', []))
        self.macro_actions = MK2MacroActionTranslator(self.button_names)
        self.use_macro_actions = self.num_players == 1 and self.action_type == 'DISCRETE'
        self.use_grouped_actions = (
            self.num_players == 1
            and self.action_type in ('FILTERED', 'MULTI_DISCRETE')
            and len(self.button_combos) > 0
        )
        self.x_span = float(MAX_X - MIN_X)
        self.x_mid = float((MIN_X + MAX_X) / 2.0)

        low = np.array([-1] * NUM_PARAMS, dtype=np.float32)
        high = np.array([1] * NUM_PARAMS, dtype=np.float32)

        if self.nn == 'CombinedPolicy':
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8),
                'scalar': spaces.Box(low, high, dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if self.use_macro_actions:
            self.action_space = spaces.Discrete(self.macro_actions.action_count())
        elif self.use_grouped_actions:
            self.action_space = spaces.MultiDiscrete([len(group) for group in self.button_combos])
        else:
            self.action_space = env.action_space

        self.last_p1_health = 0
        self.last_p2_health = 0
        self.last_p1_rounds = 0
        self.last_p2_rounds = 0
        self.last_p1_x = None
        self.last_p1_y = None
        self.last_p2_x = None
        self.last_p2_y = None
        self.last_info = None
        self.held_button_mask = 0
        self.block_hold_frames_remaining = 0
        self.button_hold_frames_remaining = 0

    def get_action_names(self):
        if self.use_grouped_actions:
            return ["VERTICAL", "HORIZONTAL_REL", "BUTTON"]
        if not self.use_macro_actions:
            return list(self.button_names)
        return self.macro_actions.get_action_names()

    def _translate_action(self, action):
        if self.use_macro_actions:
            return self.macro_actions.translate(action, self.last_info)
        if self.use_grouped_actions:
            return self._grouped_action_to_filtered(action)
        return action

    def _coerce_button_combos(self, button_combos):
        groups = []
        for group in button_combos or []:
            groups.append([int(mask) for mask in group])
        return groups

    def _bitmask_to_action_array(self, action_mask):
        action = np.zeros(len(self.button_names), dtype=np.int8)
        for button_index in range(len(self.button_names)):
            action[button_index] = (int(action_mask) >> button_index) & 1
        return action

    def _button_mask(self, button_name):
        try:
            button_index = self.button_names.index(button_name)
        except ValueError:
            return 0
        return 1 << button_index

    def _resolve_relative_horizontal_masks(self):
        left_mask = self._button_mask('LEFT')
        right_mask = self._button_mask('RIGHT')
        if not self.last_info:
            return left_mask, right_mask

        p1_x = self._safe_int(self.last_info.get('x_position'))
        p2_x = self._safe_int(self.last_info.get('enemy_x_position'))
        if p2_x >= p1_x:
            return left_mask, right_mask
        return right_mask, left_mask

    def _grouped_action_to_filtered(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.size != len(self.button_combos):
            return self._bitmask_to_action_array(0)

        action_mask = 0
        vertical_choice = int(action_arr[0])
        if 0 <= vertical_choice < len(self.button_combos[0]):
            action_mask |= self.button_combos[0][vertical_choice]

        away_mask, toward_mask = self._resolve_relative_horizontal_masks()
        horizontal_choice = int(action_arr[1])
        if horizontal_choice == 1:
            action_mask |= away_mask
        elif horizontal_choice == 2:
            action_mask |= toward_mask

        start_mask = self._button_mask('START')
        button_choice = int(action_arr[2])
        if 0 <= button_choice < len(self.button_combos[2]):
            button_mask = self.button_combos[2][button_choice]
            action_mask |= button_mask
            if start_mask and button_mask == start_mask:
                self.held_button_mask = start_mask
                self.block_hold_frames_remaining = BLOCK_HOLD_FRAMES - 1
                self.button_hold_frames_remaining = 0
            elif button_mask != 0:
                self.held_button_mask = button_mask
                self.button_hold_frames_remaining = BUTTON_HOLD_FRAMES - 1
                self.block_hold_frames_remaining = 0
        else:
            button_mask = 0

        if start_mask and button_mask == 0 and self.block_hold_frames_remaining > 0:
            action_mask |= start_mask
            self.block_hold_frames_remaining -= 1
        elif button_mask == 0 and self.button_hold_frames_remaining > 0 and self.held_button_mask != 0:
            action_mask |= self.held_button_mask
            self.button_hold_frames_remaining -= 1

        return self._bitmask_to_action_array(action_mask)

    def _build_zero_state(self):
        return np.zeros(NUM_PARAMS, dtype=np.float32)

    def _normalize_x_position(self, value):
        return _clip_unit((float(value) - self.x_mid) / (self.x_span / 2.0))

    def _normalize_y_position(self, value):
        return _clip_unit(float(value) / Y_POS_SCALE)

    def _normalize_delta_x(self, value):
        return _clip_unit(float(value) / self.x_span)

    def _normalize_delta_y(self, value):
        return _clip_unit(float(value) / Y_POS_SCALE)

    def _normalize_velocity(self, value):
        return _clip_unit(float(value) / VEL_SCALE)

    def _safe_int(self, value):
        return int(value) if value is not None else 0

    def _build_scalar_state(self, info):
        p1_health = self._safe_int(info.get('health'))
        p2_health = self._safe_int(info.get('enemy_health'))
        p1_rounds = self._safe_int(info.get('rounds_won'))
        p2_rounds = self._safe_int(info.get('enemy_rounds_won'))
        p1_x = self._safe_int(info.get('x_position'))
        p2_x = self._safe_int(info.get('enemy_x_position'))
        p1_y = self._safe_int(info.get('y_position'))
        p2_y = self._safe_int(info.get('enemy_y_position'))

        if self.last_p1_x is None:
            p1_vx = 0
            p1_vy = 0
            p2_vx = 0
            p2_vy = 0
        else:
            p1_vx = p1_x - self.last_p1_x
            p1_vy = p1_y - self.last_p1_y
            p2_vx = p2_x - self.last_p2_x
            p2_vy = p2_y - self.last_p2_y

        dx = p2_x - p1_x
        dy = p2_y - p1_y

        scalar_state = np.array([
            _clip_zero_one(p1_health / MAX_HEALTH),
            _clip_zero_one(p2_health / MAX_HEALTH),
            _clip_zero_one(p1_rounds / MAX_ROUNDS),
            _clip_zero_one(p2_rounds / MAX_ROUNDS),
            self._normalize_x_position(p1_x),
            self._normalize_y_position(p1_y),
            self._normalize_x_position(p2_x),
            self._normalize_y_position(p2_y),
            self._normalize_delta_x(dx),
            self._normalize_delta_y(dy),
            _clip_zero_one(abs(dx) / self.x_span),
            _clip_unit((p1_health - p2_health) / MAX_HEALTH),
            self._normalize_velocity(p1_vx),
            self._normalize_velocity(p1_vy),
            self._normalize_velocity(p2_vx),
            self._normalize_velocity(p2_vy),
        ], dtype=np.float32)

        self.last_p1_x = p1_x
        self.last_p1_y = p1_y
        self.last_p2_x = p2_x
        self.last_p2_y = p2_y
        return scalar_state

    def _current_obs(self, image_obs):
        if self.nn == 'CombinedPolicy':
            return {
                'image': image_obs,
                'scalar': self.state
            }
        return self.state

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        if self.use_macro_actions:
            self.macro_actions.reset()

        self.last_p1_health = self._safe_int(info.get('health'))
        self.last_p2_health = self._safe_int(info.get('enemy_health'))
        self.last_p1_rounds = self._safe_int(info.get('rounds_won'))
        self.last_p2_rounds = self._safe_int(info.get('enemy_rounds_won'))
        self.last_p1_x = None
        self.last_p1_y = None
        self.last_p2_x = None
        self.last_p2_y = None
        self.held_button_mask = 0
        self.block_hold_frames_remaining = 0
        self.button_hold_frames_remaining = 0

        if info:
            self.state = self._build_scalar_state(info)
        else:
            self.state = self._build_zero_state()
        self.last_info = info

        return self._current_obs(state), info

    def calc_reward(self, info):
        p1_health = self._safe_int(info.get('health'))
        p2_health = self._safe_int(info.get('enemy_health'))

        damage_dealt = max(0, self.last_p2_health - p2_health)
        damage_taken = max(0, self.last_p1_health - p1_health)

        rew = damage_dealt * DAMAGE_DEALT_REWARD
        rew -= damage_taken * DAMAGE_TAKEN_PENALTY

        self.last_p1_health = p1_health
        self.last_p2_health = p2_health

        return rew

    def step(self, ac):
        translated_action = self._translate_action(ac)
        ob, rew, terminated, truncated, info = self.env.step(translated_action)

        rew = self.calc_reward(info)
        self.state = self._build_scalar_state(info)
        self.last_info = info

        return self._current_obs(ob), rew, terminated, truncated, info

    def seed(self, s):
        if hasattr(self.env, 'seed'):
            return self.env.seed(s)
        return None
