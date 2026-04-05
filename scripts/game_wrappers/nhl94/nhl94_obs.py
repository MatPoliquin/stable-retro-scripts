"""
NHL94 Observation wrapper
"""

import random
import copy
from collections import deque
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_rf import register_functions
from game_wrappers.nhl94.nhl94_ai import NHL94AISystem
from game_wrappers.nhl94.nhl94_gamestate import NHL94GameState
from models_utils import load_model_for_inference


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
        random.seed(datetime.now().timestamp())

        self.num_players = num_players
        if self.selfplay_enabled and self.num_players != 2:
            raise ValueError('Self-play requires a two-player emulator environment.')
        if self.selfplay_enabled and self.action_type == 'DISCRETE':
            raise NotImplementedError('Self-play does not yet support DISCRETE action_type.')

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
        return self.env.action_space

    def _default_env_action(self):
        if self.action_type == 'MULTI_DISCRETE':
            return np.zeros(3, dtype=np.int8)
        return np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)

    def _new_action_state(self):
        return {
            'b_pressed': False,
            'c_pressed': False,
            'slapshot_frames': 0,
            'last_env_action': self._default_env_action(),
            'last_gamestate_action': [0] * 6,
        }

    def _reset_action_state(self, action_state):
        action_state['b_pressed'] = False
        action_state['c_pressed'] = False
        action_state['slapshot_frames'] = 0
        action_state['last_env_action'] = self._default_env_action()
        action_state['last_gamestate_action'] = [0] * 6

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

        self.state = tuple([0] * self.NUM_PARAMS)
        self._reset_frame_buffer()

        self.game_state = NHL94GameState(self.num_players_per_team)
        self.ram_inited = False
        self._reset_action_state(self.learner_action_state)
        self._reset_action_state(self.opponent_action_state)

        return self._get_obs(state), info

    def _process_action(self, ac, action_state):
        gamestate_ac = [0] * 6

        if isinstance(ac, (list, np.ndarray)) and len(ac) == GameConsts.INPUT_MAX:
            processed_ac = np.asarray(ac, dtype=np.int8).copy()

            if action_state['b_pressed'] and processed_ac[GameConsts.INPUT_B] == 1:
                processed_ac[GameConsts.INPUT_B] = 0
                action_state['b_pressed'] = False
            elif not action_state['b_pressed'] and processed_ac[GameConsts.INPUT_B] == 1:
                action_state['b_pressed'] = True
            else:
                action_state['b_pressed'] = False

            if processed_ac[GameConsts.INPUT_MODE] == 1:
                if action_state['slapshot_frames'] == 0:
                    action_state['slapshot_frames'] = 1
                    processed_ac[GameConsts.INPUT_C] = 1
                else:
                    action_state['slapshot_frames'] += 1
                    processed_ac[GameConsts.INPUT_C] = 1
                    if action_state['slapshot_frames'] >= 60:
                        action_state['slapshot_frames'] = 0
                        processed_ac[GameConsts.INPUT_C] = 0
            else:
                if action_state['c_pressed'] and processed_ac[GameConsts.INPUT_C] == 1:
                    processed_ac[GameConsts.INPUT_C] = 0
                    action_state['c_pressed'] = False
                elif not action_state['c_pressed'] and processed_ac[GameConsts.INPUT_C] == 1:
                    action_state['c_pressed'] = True
                else:
                    action_state['c_pressed'] = False
                action_state['slapshot_frames'] = 0

            gamestate_ac[0] = bool(processed_ac[GameConsts.INPUT_UP])
            gamestate_ac[1] = bool(processed_ac[GameConsts.INPUT_DOWN])
            gamestate_ac[2] = bool(processed_ac[GameConsts.INPUT_LEFT])
            gamestate_ac[3] = bool(processed_ac[GameConsts.INPUT_RIGHT])
            gamestate_ac[4] = bool(processed_ac[GameConsts.INPUT_B])
            gamestate_ac[5] = bool(processed_ac[GameConsts.INPUT_C])

        elif isinstance(ac, (list, np.ndarray)) and len(ac) == 3:
            processed_ac = np.asarray(ac, dtype=np.int8).copy()

            if processed_ac[2] == 1:
                if action_state['b_pressed']:
                    processed_ac[2] = 0
                    action_state['b_pressed'] = False
                else:
                    action_state['b_pressed'] = True
            else:
                action_state['b_pressed'] = False

            if processed_ac[2] == 2:
                if action_state['slapshot_frames'] == 0:
                    action_state['slapshot_frames'] = 1
                else:
                    action_state['slapshot_frames'] += 1
                    if action_state['slapshot_frames'] >= 60:
                        processed_ac[2] = 0
                        action_state['slapshot_frames'] = 0
            else:
                action_state['slapshot_frames'] = 0

            gamestate_ac[0] = processed_ac[0] == 1
            gamestate_ac[1] = processed_ac[0] == 2
            gamestate_ac[2] = processed_ac[1] == 1
            gamestate_ac[3] = processed_ac[1] == 2
            gamestate_ac[4] = processed_ac[2] == 1
            gamestate_ac[5] = processed_ac[2] == 2
        else:
            raise ValueError(f"Unsupported action format: {ac}")

        action_state['last_env_action'] = processed_ac.copy()
        action_state['last_gamestate_action'] = list(gamestate_ac)
        return processed_ac, gamestate_ac

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

        ac2 = learner_action
        if self.num_players == 2:
            if self.selfplay_enabled:
                opponent_action = self._compute_opponent_action()
                opponent_action, _ = self._process_action(opponent_action, self.opponent_action_state)
                self.opponent_input_overide(opponent_action)
            else:
                opponent_action = np.zeros_like(learner_action)

            ac2 = np.concatenate([learner_action, opponent_action])

        ob, rew, terminated, truncated, info = self.env.step(ac2)

        if not self.ram_inited:
            self.init_function(self.env, self.env_name)
            self.ram_inited = True

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
