"""
NHL94 Observation wrapper
"""

import copy
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_intents import (
    HOCKEY_INTENT_DPAD_ACTIONS,
    HOCKEY_INTENT_DPAD_ACTION_SPACE,
    HOCKEY_INTENT_CARRY_PUCK,
    HOCKEY_INTENT_NORMAL_SHOOT,
    HOCKEY_INTENT_SLAPSHOT,
    HOCKEY_INTENT_ONE_TIMER,
    HOCKEY_INTENT_POKE_CHECK,
    HOCKEY_INTENT_CHANGE_PLAYER,
    HOCKEY_INTENT_CATCH_PUCK,
    HOCKEY_INTENT_PASS_START,
)
from game_wrappers.nhl94.nhl94_rf import register_functions
from game_wrappers.nhl94.nhl94_ai import NHL94AISystem
from game_wrappers.nhl94.nhl94_gamestate import NHL94GameState
from models_utils import load_model_for_inference


HOCKEY_PASS_RELEASE_FRAMES = 5
HOCKEY_ONE_TIMER_SHOT_DELAY_FRAMES = 1
HOCKEY_ONE_TIMER_SHOT_FRAMES = 12
HOCKEY_SLAPSHOT_HOLD_FRAMES = 18
HOCKEY_POKE_DISTANCE = 18.0
HOCKEY_CATCH_MAX_LOOKAHEAD_FRAMES = 36
HOCKEY_CATCH_PLAYER_SPEED_ESTIMATE = 7.2
HOCKEY_CATCH_BURST_DISTANCE = 22.0


class NHL94Observation2PEnv(gym.Wrapper):
    def __init__(self, env, args, num_players, rf_name):
        gym.Wrapper.__init__(self, env)

        self.nn = args.nn
        self.env_name = args.env
        self.action_type = getattr(args, 'action_type', 'FILTERED').upper()
        self.selfplay_enabled = bool(getattr(args, 'selfplay', False))
        self.selfplay_role = self._resolve_selfplay_role(args, rf_name)
        self.deterministic = bool(getattr(args, 'deterministic', True))
        self.opponent_model_alg = getattr(args, 'alg', 'ppo2')

        self.rf_name = rf_name
        self.init_function, self.reward_function, self.done_function, self.init_model, self.set_model_input, self.input_overide = register_functions(self.rf_name)
        self.opponent_model = None
        self.opponent_model_path = ''
        self.opponent_set_model_input = None
        self.opponent_input_overide = None

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

        self.num_players = num_players
        if self.selfplay_enabled and self.num_players != 2:
            raise ValueError('Self-play requires a two-player emulator environment.')
        if self.selfplay_enabled and self.action_type == 'DISCRETE':
            raise NotImplementedError('Self-play does not yet support DISCRETE action_type.')
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            if self.env_name != 'NHL94-Genesis-v0' or self.num_players_per_team != 5:
                raise ValueError('HOCKEY_INTENT_DPAD is only supported for NHL94-Genesis-v0 full 5v5.')
            if self.num_players != 1:
                raise ValueError('HOCKEY_INTENT_DPAD currently supports one learner controller only.')

        self.action_space = self._build_action_space()

        self.ai_sys = NHL94AISystem(args, env, None)

        self.ram_inited = False
        self.learner_action_state = self._new_action_state()
        self.opponent_action_state = self._new_action_state()

        if self.selfplay_enabled:
            self.set_selfplay_role(self.selfplay_role)
            self.set_opponent_model(getattr(args, 'load_opponent_model', ''))

    def _resolve_selfplay_role(self, args, rf_name):
        if rf_name == 'SelfPlayDefenseFinetune':
            return 'defense'
        if rf_name == 'SelfPlayOffenseFinetune':
            return 'offense'
        return getattr(args, 'selfplay_role', 'offense')

    def _build_action_space(self):
        if self.action_type == 'FILTERED':
            return gym.spaces.MultiBinary(GameConsts.INPUT_MAX)
        if self.action_type == 'MULTI_DISCRETE':
            return gym.spaces.MultiDiscrete([3, 3, 3])
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            return gym.spaces.MultiDiscrete(HOCKEY_INTENT_DPAD_ACTION_SPACE)
        return self.env.action_space

    def _default_env_action(self):
        if self.action_type == 'MULTI_DISCRETE':
            return np.zeros(3, dtype=np.int8)
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            return np.zeros(len(HOCKEY_INTENT_DPAD_ACTION_SPACE), dtype=np.int8)
        return np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)

    def _default_controller_action(self):
        return np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)

    def _default_reset_action(self):
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            return self._default_controller_action()
        return self._default_env_action()

    def get_action_names(self):
        if self.action_type == 'HOCKEY_INTENT_DPAD':
            return ['INTENT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOOST']
        return getattr(self.env, 'buttons', [])

    def _new_action_state(self):
        return {
            'slapshot_frames': 0,
            'last_env_action': self._default_env_action(),
            'last_gamestate_action': [0] * 6,
            'hockey_macro': 'none',
            'pass_release_frames': 0,
            'one_timer_shot_delay': 0,
            'one_timer_shot_frames': 0,
            'slapshot_charge_frames': 0,
        }

    def _reset_action_state(self, action_state):
        action_state['slapshot_frames'] = 0
        action_state['last_env_action'] = self._default_env_action()
        action_state['last_gamestate_action'] = [0] * 6
        action_state['hockey_macro'] = 'none'
        action_state['pass_release_frames'] = 0
        action_state['one_timer_shot_delay'] = 0
        action_state['one_timer_shot_frames'] = 0
        action_state['slapshot_charge_frames'] = 0

    def set_selfplay_role(self, role):
        if role not in ('offense', 'defense'):
            raise ValueError(f"Unsupported self-play role: {role}")

        self.selfplay_role = role
        opponent_rf = 'DefenseZone' if role == 'offense' else 'ScoreGoal'
        _, _, _, _, self.opponent_set_model_input, self.opponent_input_overide = register_functions(opponent_rf)
        return self.selfplay_role

    def set_opponent_model(self, path):
        if not path:
            self.opponent_model = None
            self.opponent_model_path = ''
            return None

        if path == self.opponent_model_path and self.opponent_model is not None:
            return self.opponent_model_path

        self.opponent_model = load_model_for_inference(path, self.opponent_model_alg)
        self.opponent_model_path = path
        return self.opponent_model_path

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

        self.init_function(self.env, self.env_name)

        reset_action = self._default_reset_action()
        if self.num_players == 2:
            reset_action = np.concatenate([reset_action, reset_action])

        state, _, _, _, info = self.env.step(reset_action)

        self.game_state = NHL94GameState(self.num_players_per_team)
        self.ram_inited = True
        self._reset_action_state(self.learner_action_state)
        self._reset_action_state(self.opponent_action_state)

        self.game_state.BeginFrame(info, [0] * 6)
        self.game_state.EndFrame()
        self.state = self.set_model_input(self.game_state)
        self._reset_frame_buffer()
        if self.uses_sequence_obs:
            self.frame_buffer.append(self._get_scalar_state_array().copy())

        return self._get_obs(state), info

    def _team_has_puck(self, team):
        return bool(getattr(team, 'player_haspuck', False) or getattr(team, 'goalie_haspuck', False))

    def _player_has_puck(self, team):
        return bool(getattr(team, 'player_haspuck', False))

    def _opponent_has_puck(self):
        return self._team_has_puck(self.game_state.team2)

    def _hockey_controlled_player(self, team):
        return team.get_controlled_player()

    def _hockey_distance(self, first, second):
        return GameConsts.Distance(
            (getattr(first, 'x', 0) or 0, getattr(first, 'y', 0) or 0),
            (getattr(second, 'x', 0) or 0, getattr(second, 'y', 0) or 0),
        )

    def _hockey_distance_xy(self, first_x, first_y, second_x, second_y):
        return GameConsts.Distance((first_x, first_y), (second_x, second_y))

    def _set_controller_gamestate(self, processed_ac, gamestate_ac):
        gamestate_ac[0] = bool(processed_ac[GameConsts.INPUT_UP])
        gamestate_ac[1] = bool(processed_ac[GameConsts.INPUT_DOWN])
        gamestate_ac[2] = bool(processed_ac[GameConsts.INPUT_LEFT])
        gamestate_ac[3] = bool(processed_ac[GameConsts.INPUT_RIGHT])
        gamestate_ac[4] = bool(processed_ac[GameConsts.INPUT_B])
        gamestate_ac[5] = bool(processed_ac[GameConsts.INPUT_C])

    def _steer_hockey_toward(self, processed_ac, player, target, deadzone=4):
        target_x, target_y = target
        delta_x = target_x - (getattr(player, 'x', 0) or 0)
        delta_y = target_y - (getattr(player, 'y', 0) or 0)

        if delta_x <= -deadzone:
            processed_ac[GameConsts.INPUT_LEFT] = 1
            processed_ac[GameConsts.INPUT_RIGHT] = 0
        elif delta_x >= deadzone:
            processed_ac[GameConsts.INPUT_RIGHT] = 1
            processed_ac[GameConsts.INPUT_LEFT] = 0

        if delta_y <= -deadzone:
            processed_ac[GameConsts.INPUT_DOWN] = 1
            processed_ac[GameConsts.INPUT_UP] = 0
        elif delta_y >= deadzone:
            processed_ac[GameConsts.INPUT_UP] = 1
            processed_ac[GameConsts.INPUT_DOWN] = 0

    def _hockey_predict_puck_position(self, puck_x, puck_y, puck_vx, puck_vy, frames):
        predicted_x = puck_x + puck_vx * frames * 0.45
        predicted_y = puck_y + puck_vy * frames * 0.45

        if predicted_x < -GameConsts.MAX_PLAYER_X:
            predicted_x = -GameConsts.MAX_PLAYER_X + (-GameConsts.MAX_PLAYER_X - predicted_x) * 0.65
        elif predicted_x > GameConsts.MAX_PLAYER_X:
            predicted_x = GameConsts.MAX_PLAYER_X - (predicted_x - GameConsts.MAX_PLAYER_X) * 0.65

        if predicted_y < -GameConsts.MAX_PLAYER_Y:
            predicted_y = -GameConsts.MAX_PLAYER_Y + (-GameConsts.MAX_PLAYER_Y - predicted_y) * 0.45
        elif predicted_y > GameConsts.MAX_PLAYER_Y:
            predicted_y = GameConsts.MAX_PLAYER_Y - (predicted_y - GameConsts.MAX_PLAYER_Y) * 0.45

        return predicted_x, predicted_y

    def _hockey_opponent_puck_carrier(self):
        opponents = self.game_state.team2
        if self._team_has_puck(opponents):
            return opponents.get_controlled_player()
        return None

    def _hockey_carrier_stick_target(self, carrier, puck):
        carrier_x = getattr(carrier, 'x', 0) or 0
        carrier_y = getattr(carrier, 'y', 0) or 0
        puck_x = getattr(puck, 'x', 0) or 0
        puck_y = getattr(puck, 'y', 0) or 0
        toward_puck_x = puck_x - carrier_x
        toward_puck_y = puck_y - carrier_y
        length = max(1.0, self._hockey_distance_xy(carrier_x, carrier_y, puck_x, puck_y))
        stick_x = carrier_x + toward_puck_x / length * 10.0
        stick_y = carrier_y + toward_puck_y / length * 10.0
        return (
            stick_x + (getattr(carrier, 'vx', 0) or 0) * 0.65,
            stick_y + (getattr(carrier, 'vy', 0) or 0) * 0.65,
        )

    def _hockey_choose_intercept_target(self, controlled):
        puck = self.game_state.puck
        player_x = getattr(controlled, 'x', 0) or 0
        player_y = getattr(controlled, 'y', 0) or 0
        player_vx = getattr(controlled, 'vx', 0) or 0
        player_vy = getattr(controlled, 'vy', 0) or 0
        puck_x = getattr(puck, 'x', 0) or 0
        puck_y = getattr(puck, 'y', 0) or 0
        puck_vx = getattr(puck, 'vx', 0) or 0
        puck_vy = getattr(puck, 'vy', 0) or 0

        best_target = (puck_x, puck_y)
        best_error = float('inf')
        for frames in range(1, HOCKEY_CATCH_MAX_LOOKAHEAD_FRAMES + 1):
            predicted_puck_x, predicted_puck_y = self._hockey_predict_puck_position(
                puck_x, puck_y, puck_vx, puck_vy, frames,
            )
            desired_player_x = player_x + player_vx * min(frames, 8) * 0.20
            desired_player_y = player_y + player_vy * min(frames, 8) * 0.20
            distance = self._hockey_distance_xy(
                desired_player_x, desired_player_y, predicted_puck_x, predicted_puck_y,
            )
            reachable_distance = HOCKEY_CATCH_PLAYER_SPEED_ESTIMATE * frames
            error = abs(distance - reachable_distance) + frames * 0.15
            if error < best_error:
                best_error = error
                best_target = (predicted_puck_x, predicted_puck_y)

        carrier = self._hockey_opponent_puck_carrier()
        if carrier is not None:
            carrier_target = self._hockey_carrier_stick_target(carrier, puck)
            carrier_weight = 0.65 if self._hockey_distance(controlled, carrier) < 42 else 0.35
            best_target = (
                best_target[0] * (1.0 - carrier_weight) + carrier_target[0] * carrier_weight,
                best_target[1] * (1.0 - carrier_weight) + carrier_target[1] * carrier_weight,
            )

        return (
            int(np.clip(best_target[0], -GameConsts.MAX_PLAYER_X, GameConsts.MAX_PLAYER_X)),
            int(np.clip(best_target[1], -GameConsts.MAX_PLAYER_Y, GameConsts.MAX_PLAYER_Y)),
        )

    def _hockey_closing_speed(self, controlled):
        puck = self.game_state.puck
        delta_x = (getattr(puck, 'x', 0) or 0) - (getattr(controlled, 'x', 0) or 0)
        delta_y = (getattr(puck, 'y', 0) or 0) - (getattr(controlled, 'y', 0) or 0)
        distance = max(1.0, self._hockey_distance_xy(0, 0, delta_x, delta_y))
        relative_vx = (getattr(controlled, 'vx', 0) or 0) - (getattr(puck, 'vx', 0) or 0)
        relative_vy = (getattr(controlled, 'vy', 0) or 0) - (getattr(puck, 'vy', 0) or 0)
        return (relative_vx * delta_x + relative_vy * delta_y) / distance

    def _apply_hockey_catch_puck(self, processed_ac, action_state, controlled):
        target = self._hockey_choose_intercept_target(controlled)
        self._steer_hockey_toward(processed_ac, controlled, target, deadzone=4)

        puck = self.game_state.puck
        distance_to_target = self._hockey_distance_xy(
            getattr(controlled, 'x', 0) or 0,
            getattr(controlled, 'y', 0) or 0,
            target[0],
            target[1],
        )
        distance_to_puck = self._hockey_distance(controlled, puck)
        closing_speed = self._hockey_closing_speed(controlled)

        if distance_to_puck > 10 and (distance_to_target > HOCKEY_CATCH_BURST_DISTANCE or (closing_speed < 1.5 and distance_to_puck > 14)):
            processed_ac[GameConsts.INPUT_C] = 1

        action_state['hockey_macro'] = 'catch_puck'

    def _hockey_pass_target(self, intent):
        target_index = intent - HOCKEY_INTENT_PASS_START
        players = list(getattr(self.game_state.team1, 'players', []) or [])
        controlled_index = getattr(self.game_state.team1, 'control', 0) - 1
        teammates = [player for index, player in enumerate(players) if index != controlled_index]
        if target_index < 0 or target_index >= len(teammates):
            return None
        return teammates[target_index]

    def _best_one_timer_target(self):
        team = self.game_state.team1
        opponents = self.game_state.team2
        controlled = self._hockey_controlled_player(team)
        controlled_index = getattr(team, 'control', 0) - 1
        opponent_group = list(getattr(opponents, 'players', []) or []) + [getattr(opponents, 'goalie', None)]
        opponent_group = [player for player in opponent_group if player is not None]

        best_player = None
        best_score = -float('inf')
        for index, player in enumerate(team.players):
            if index == controlled_index:
                continue
            if getattr(player, 'is_falling', 0.0) or getattr(player, 'is_dive', 0.0):
                continue

            receiver_space = min(
                [self._hockey_distance(player, opponent) for opponent in opponent_group] or [999.0]
            )
            lateral_separation = abs((getattr(player, 'x', 0) or 0) - (getattr(controlled, 'x', 0) or 0))
            vertical_separation = abs((getattr(player, 'y', 0) or 0) - (getattr(controlled, 'y', 0) or 0))
            score = receiver_space + lateral_separation * 0.45 + vertical_separation * 0.20
            if getattr(player, 'passing_lane_clear', False):
                score += 24.0
            if abs(getattr(player, 'x', 0) or 0) > 96:
                score -= 18.0

            if score > best_score:
                best_score = score
                best_player = player

        return best_player

    def _apply_pending_hockey_macro(self, processed_ac, action_state):
        if self._opponent_has_puck():
            action_state['one_timer_shot_delay'] = 0
            action_state['one_timer_shot_frames'] = 0

        if action_state['pass_release_frames'] > 0:
            action_state['pass_release_frames'] -= 1

        if action_state['one_timer_shot_frames'] <= 0:
            return False

        action_state['hockey_macro'] = 'one_timer_wait'
        if action_state['one_timer_shot_delay'] > 0:
            action_state['one_timer_shot_delay'] -= 1
            return True

        processed_ac[GameConsts.INPUT_C] = 1
        action_state['one_timer_shot_frames'] -= 1
        action_state['hockey_macro'] = 'one_timer_shoot'
        return True

    def _apply_hockey_intent(self, intent_ac, processed_ac, gamestate_ac, action_state):
        action_state['hockey_macro'] = 'none'

        if self._apply_pending_hockey_macro(processed_ac, action_state):
            self._set_controller_gamestate(processed_ac, gamestate_ac)
            return

        intent = int(np.clip(intent_ac[0], 0, len(HOCKEY_INTENT_DPAD_ACTIONS) - 1))
        boost = bool(intent_ac[5])
        team = self.game_state.team1
        controlled = self._hockey_controlled_player(team)
        own_player_has_puck = self._player_has_puck(team)
        own_team_has_puck = self._team_has_puck(team)

        if intent == HOCKEY_INTENT_CARRY_PUCK:
            if own_player_has_puck:
                action_state['hockey_macro'] = 'carry_puck'

        elif intent == HOCKEY_INTENT_NORMAL_SHOOT:
            if own_player_has_puck:
                processed_ac[GameConsts.INPUT_C] = 1
                action_state['hockey_macro'] = 'normal_shoot'

        elif intent == HOCKEY_INTENT_SLAPSHOT:
            if own_player_has_puck and action_state['slapshot_charge_frames'] < HOCKEY_SLAPSHOT_HOLD_FRAMES:
                processed_ac[GameConsts.INPUT_C] = 1
                action_state['slapshot_charge_frames'] += 1
                action_state['hockey_macro'] = 'slapshot_charge'
            else:
                if action_state['slapshot_charge_frames'] > 0:
                    action_state['hockey_macro'] = 'slapshot_release'
                action_state['slapshot_charge_frames'] = 0
        else:
            action_state['slapshot_charge_frames'] = 0

        if intent == HOCKEY_INTENT_POKE_CHECK:
            distance_to_puck = self._hockey_distance(controlled, self.game_state.puck)
            if not own_team_has_puck and (self._opponent_has_puck() or distance_to_puck <= HOCKEY_POKE_DISTANCE):
                processed_ac[GameConsts.INPUT_B] = 1
                action_state['hockey_macro'] = 'poke_check'

        elif intent == HOCKEY_INTENT_CHANGE_PLAYER:
            if not own_team_has_puck:
                processed_ac[GameConsts.INPUT_B] = 1
                action_state['hockey_macro'] = 'change_player'

        elif intent == HOCKEY_INTENT_CATCH_PUCK:
            if not own_team_has_puck:
                self._apply_hockey_catch_puck(processed_ac, action_state, controlled)

        elif intent >= HOCKEY_INTENT_PASS_START:
            target = self._hockey_pass_target(intent)
            if target is not None:
                self._steer_hockey_toward(processed_ac, controlled, (target.x, target.y), deadzone=3)
                if own_team_has_puck and action_state['pass_release_frames'] <= 0:
                    processed_ac[GameConsts.INPUT_B] = 1
                    action_state['pass_release_frames'] = HOCKEY_PASS_RELEASE_FRAMES
                    action_state['hockey_macro'] = HOCKEY_INTENT_DPAD_ACTIONS[intent].lower()

        elif intent == HOCKEY_INTENT_ONE_TIMER:
            target = self._best_one_timer_target()
            if target is not None:
                self._steer_hockey_toward(processed_ac, controlled, (target.x, target.y), deadzone=3)
                if own_team_has_puck and action_state['pass_release_frames'] <= 0:
                    processed_ac[GameConsts.INPUT_B] = 1
                    action_state['pass_release_frames'] = HOCKEY_PASS_RELEASE_FRAMES
                    action_state['one_timer_shot_delay'] = HOCKEY_ONE_TIMER_SHOT_DELAY_FRAMES
                    action_state['one_timer_shot_frames'] = HOCKEY_ONE_TIMER_SHOT_FRAMES
                    action_state['hockey_macro'] = 'one_timer_pass'

        if boost and not own_team_has_puck and not processed_ac[GameConsts.INPUT_C]:
            processed_ac[GameConsts.INPUT_C] = 1
            if action_state['hockey_macro'] == 'none':
                action_state['hockey_macro'] = 'boost'

        self._set_controller_gamestate(processed_ac, gamestate_ac)

    def _process_action(self, ac, action_state):
        gamestate_ac = [0] * 6

        if isinstance(ac, (list, np.ndarray)) and len(ac) == GameConsts.INPUT_MAX:
            processed_ac = np.asarray(ac, dtype=np.int8).copy()
            recorded_ac = processed_ac

            gamestate_ac[0] = bool(processed_ac[GameConsts.INPUT_UP])
            gamestate_ac[1] = bool(processed_ac[GameConsts.INPUT_DOWN])
            gamestate_ac[2] = bool(processed_ac[GameConsts.INPUT_LEFT])
            gamestate_ac[3] = bool(processed_ac[GameConsts.INPUT_RIGHT])
            gamestate_ac[4] = bool(processed_ac[GameConsts.INPUT_B])
            gamestate_ac[5] = bool(processed_ac[GameConsts.INPUT_C])

        elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
            processed_ac = np.asarray(ac, dtype=np.int8).copy()
            recorded_ac = processed_ac

            gamestate_ac[0] = processed_ac[0] == 1
            gamestate_ac[1] = processed_ac[0] == 2
            gamestate_ac[2] = processed_ac[1] == 1
            gamestate_ac[3] = processed_ac[1] == 2
            gamestate_ac[4] = processed_ac[2] == 1
            gamestate_ac[5] = processed_ac[2] == 2

        elif self.action_type == 'HOCKEY_INTENT_DPAD' and isinstance(ac, (list, np.ndarray)) and len(ac) == 6:
            intent_ac = np.asarray(ac, dtype=np.int8).copy()
            recorded_ac = intent_ac
            processed_ac = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)

            up = bool(intent_ac[1])
            down = bool(intent_ac[2])
            left = bool(intent_ac[3])
            right = bool(intent_ac[4])

            if up != down:
                processed_ac[GameConsts.INPUT_UP if up else GameConsts.INPUT_DOWN] = 1
            if left != right:
                processed_ac[GameConsts.INPUT_LEFT if left else GameConsts.INPUT_RIGHT] = 1

            self._apply_hockey_intent(intent_ac, processed_ac, gamestate_ac, action_state)
        else:
            raise ValueError(f"Unsupported action format: {ac}")

        if gamestate_ac[5]:
            action_state['slapshot_frames'] += 1
        else:
            action_state['slapshot_frames'] = 0
        action_state['last_env_action'] = recorded_ac.copy()
        action_state['last_gamestate_action'] = list(gamestate_ac)
        return processed_ac, gamestate_ac

    def _gamestate_action_from_env_action(self, env_action):
        gamestate_ac = [0] * 6

        if isinstance(env_action, (list, np.ndarray)) and len(env_action) == GameConsts.INPUT_MAX:
            env_action = np.asarray(env_action, dtype=np.int8)
            gamestate_ac[0] = bool(env_action[GameConsts.INPUT_UP])
            gamestate_ac[1] = bool(env_action[GameConsts.INPUT_DOWN])
            gamestate_ac[2] = bool(env_action[GameConsts.INPUT_LEFT])
            gamestate_ac[3] = bool(env_action[GameConsts.INPUT_RIGHT])
            gamestate_ac[4] = bool(env_action[GameConsts.INPUT_B])
            gamestate_ac[5] = bool(env_action[GameConsts.INPUT_C])
            return gamestate_ac

        if isinstance(env_action, (list, np.ndarray)) and len(env_action) == 3:
            env_action = np.asarray(env_action, dtype=np.int8)
            gamestate_ac[0] = env_action[0] == 1
            gamestate_ac[1] = env_action[0] == 2
            gamestate_ac[2] = env_action[1] == 1
            gamestate_ac[3] = env_action[1] == 2
            gamestate_ac[4] = env_action[2] == 1
            gamestate_ac[5] = env_action[2] == 2
            return gamestate_ac

        raise ValueError(f"Unsupported env action format: {env_action}")

    def _finalize_masked_action(self, env_action, action_state):
        masked_gamestate_ac = self._gamestate_action_from_env_action(env_action)

        if not masked_gamestate_ac[5]:
            action_state['slapshot_frames'] = 0
            action_state['slapshot_charge_frames'] = 0
            action_state['one_timer_shot_delay'] = 0
            action_state['one_timer_shot_frames'] = 0
            if action_state['hockey_macro'] in {
                'normal_shoot',
                'slapshot_charge',
                'slapshot_release',
                'one_timer_wait',
                'one_timer_shoot',
                'boost',
                'catch_puck',
            }:
                action_state['hockey_macro'] = 'none'

        action_state['last_env_action'] = np.asarray(env_action, dtype=np.int8).copy()
        action_state['last_gamestate_action'] = list(masked_gamestate_ac)
        return masked_gamestate_ac

    def _mirror_env_action(self, ac):
        mirrored = np.asarray(ac, dtype=np.int8).copy()
        if len(mirrored) == GameConsts.INPUT_MAX:
            up = mirrored[GameConsts.INPUT_UP]
            down = mirrored[GameConsts.INPUT_DOWN]
            left = mirrored[GameConsts.INPUT_LEFT]
            right = mirrored[GameConsts.INPUT_RIGHT]
            mirrored[GameConsts.INPUT_UP] = down
            mirrored[GameConsts.INPUT_DOWN] = up
            mirrored[GameConsts.INPUT_LEFT] = right
            mirrored[GameConsts.INPUT_RIGHT] = left
            return mirrored

        if len(mirrored) == 3:
            if mirrored[0] == 1:
                mirrored[0] = 2
            elif mirrored[0] == 2:
                mirrored[0] = 1

            if mirrored[1] == 1:
                mirrored[1] = 2
            elif mirrored[1] == 2:
                mirrored[1] = 1
            return mirrored

        raise ValueError(f"Unsupported action format for mirroring: {ac}")

    def _mirror_gamestate_action(self, gamestate_action):
        mirrored = list(gamestate_action)
        mirrored[0], mirrored[1] = mirrored[1], mirrored[0]
        mirrored[2], mirrored[3] = mirrored[3], mirrored[2]
        return mirrored

    def _should_flip_zones_for_opponent(self):
        period = int(getattr(self.game_state, 'period', 1) or 1)
        return (period % 2) == 0

    def _rotate_player_180(self, player):
        for attr in (
            'x', 'y', 'vx', 'vy',
            'rel_puck_x', 'rel_puck_y', 'rel_puck_vx', 'rel_puck_vy',
            'rel_controlled_x', 'rel_controlled_y', 'rel_controlled_vx', 'rel_controlled_vy',
            'ori_x', 'ori_y',
        ):
            if hasattr(player, attr):
                setattr(player, attr, -getattr(player, attr))

        if hasattr(player, 'orientation'):
            player.orientation = (player.orientation + 4) % 8

    def _rotate_team_180(self, team):
        for player in list(team.players) + [team.goalie]:
            self._rotate_player_180(player)

        team.stats.fullstar_x = -team.stats.fullstar_x
        team.stats.fullstar_y = -team.stats.fullstar_y
        team.stats.emptystar_x = -team.stats.emptystar_x
        team.stats.emptystar_y = -team.stats.emptystar_y

    def _recompute_team_relationships(self, team, puck):
        controlled_idx = max(0, team.control - 1) if team.control > 0 else 0
        controlled_player = team.goalie if team.control == 0 else team.players[controlled_idx]

        for player in team.players:
            player.rel_puck_x = puck.x - player.x
            player.rel_puck_y = puck.y - player.y
            player.rel_puck_vx = puck.vx - player.vx
            player.rel_puck_vy = puck.vy - player.vy
            player.rel_controlled_x = player.x - controlled_player.x
            player.rel_controlled_y = player.y - controlled_player.y
            player.rel_controlled_vx = player.vx - controlled_player.vx
            player.rel_controlled_vy = player.vy - controlled_player.vy
            player.dist_to_controlled = GameConsts.Distance(
                (player.x, player.y),
                (controlled_player.x, controlled_player.y),
            )
            player.dist_to_puck = GameConsts.Distance(
                (player.x, player.y),
                (puck.x, puck.y),
            )

        team.goalie.rel_puck_x = puck.x - team.goalie.x
        team.goalie.rel_puck_y = puck.y - team.goalie.y
        team.goalie.rel_puck_vx = puck.vx - team.goalie.vx
        team.goalie.rel_puck_vy = puck.vy - team.goalie.vy
        team.goalie.rel_controlled_x = team.goalie.x - controlled_player.x
        team.goalie.rel_controlled_y = team.goalie.y - controlled_player.y
        team.goalie.rel_controlled_vx = team.goalie.vx - controlled_player.vx
        team.goalie.rel_controlled_vy = team.goalie.vy - controlled_player.vy
        team.goalie.dist_to_controlled = GameConsts.Distance(
            (team.goalie.x, team.goalie.y),
            (controlled_player.x, controlled_player.y),
        )
        team.goalie.dist_to_puck = GameConsts.Distance(
            (team.goalie.x, team.goalie.y),
            (puck.x, puck.y),
        )

    def _refresh_derived_state(self, mirrored_state):
        self._recompute_team_relationships(mirrored_state.team1, mirrored_state.puck)
        self._recompute_team_relationships(mirrored_state.team2, mirrored_state.puck)
        mirrored_state.update_nets()
        mirrored_state._update_passing_lanes()
        mirrored_state._update_opponent_controlled_distances()
        mirrored_state.team1.Normalize()
        mirrored_state.team2.Normalize()
        mirrored_state.nz_puck.x = mirrored_state.puck.x / GameConsts.MAX_PUCK_X
        mirrored_state.nz_puck.y = mirrored_state.puck.y / GameConsts.MAX_PUCK_Y
        mirrored_state.nz_puck.vx = mirrored_state.puck.vx / GameConsts.MAX_VEL_XY
        mirrored_state.nz_puck.vy = mirrored_state.puck.vy / GameConsts.MAX_VEL_XY

    def _build_opponent_view_state(self):
        mirrored_state = copy.deepcopy(self.game_state)
        mirrored_state.Flip()

        mirrored_state.team1.controller = 1
        mirrored_state.team1.ram_var_prefix = 'p1_'
        mirrored_state.team1.ram_var_goalie_prefix = 'g1_'
        mirrored_state.team2.controller = 2
        mirrored_state.team2.ram_var_prefix = 'p2_'
        mirrored_state.team2.ram_var_goalie_prefix = 'g2_'

        if self._should_flip_zones_for_opponent():
            self._rotate_team_180(mirrored_state.team1)
            self._rotate_team_180(mirrored_state.team2)
            self._rotate_player_180(mirrored_state.puck)
            mirrored_state.action = self._mirror_gamestate_action(self.opponent_action_state['last_gamestate_action'])
        else:
            mirrored_state.action = list(self.opponent_action_state['last_gamestate_action'])
        mirrored_state.slapshot_frames_held = self.opponent_action_state['slapshot_frames']

        self._refresh_derived_state(mirrored_state)
        return mirrored_state

    def _compute_opponent_action(self):
        if self.opponent_model is None:
            if self.opponent_model_path:
                self.set_opponent_model(self.opponent_model_path)
            if self.opponent_model is None:
                raise ValueError('Self-play requires load_opponent_model to be set.')

        opponent_state = self._build_opponent_view_state()
        opponent_obs = self.opponent_set_model_input(opponent_state)
        opponent_action, _ = self.opponent_model.predict(opponent_obs, deterministic=self.deterministic)
        if self._should_flip_zones_for_opponent():
            opponent_action = self._mirror_env_action(opponent_action)
        self.opponent_input_overide(opponent_action)
        return opponent_action

    def step(self, ac):
        learner_action, gamestate_ac = self._process_action(ac, self.learner_action_state)
        self.input_overide(learner_action)
        gamestate_ac = self._finalize_masked_action(learner_action, self.learner_action_state)

        ac2 = learner_action
        if self.num_players == 2:
            if self.selfplay_enabled:
                opponent_action = self._compute_opponent_action()
                opponent_action, _ = self._process_action(opponent_action, self.opponent_action_state)
                self.opponent_input_overide(opponent_action)
                self._finalize_masked_action(opponent_action, self.opponent_action_state)
            else:
                opponent_action = np.zeros_like(learner_action)

            ac2 = np.concatenate([learner_action, opponent_action])

        ob, rew, terminated, truncated, info = self.env.step(ac2)

        self.game_state.BeginFrame(info, gamestate_ac)

        rew = self.reward_function(self.game_state)
        terminated = self.done_function(self.game_state)

        self.game_state.EndFrame()

        self.state = self.set_model_input(self.game_state)
        if self.uses_sequence_obs:
            self.frame_buffer.append(self._get_scalar_state_array().copy())

        return self._get_obs(ob), rew, terminated, truncated, info

    def seed(self, s):
        if hasattr(self.env, 'seed'):
            return self.env.seed(s)
        return None
