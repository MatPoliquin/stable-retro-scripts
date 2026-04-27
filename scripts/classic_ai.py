import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from game_wrappers.nhl94.nhl94_const import GameConsts


class ClassicAIModel:
    """Inference-only scripted controller for NHL94.

    The first implementation targets the NHL94 game-state wrapper directly.
    It exposes a model-like interface so existing play tooling can select it
    with ``--nn=ClassicAI``.
    """

    def __init__(self, args=None, env=None):
        self.args = args
        self.env = env
        self._rng = random.Random()
        self._last_action_preferences = np.zeros((1, GameConsts.INPUT_MAX), dtype=np.float32)
        self._attack_lane_bias = self._rng.choice((-1, 1))
        self._attack_lane_frames = 0
        self._last_possession_control = None
        self._last_progress_score = None
        self._possession_frames = 0
        self._stall_frames = 0
        self._attack_plan_index = -1
        self._current_attack_plan = 0
        self._pending_one_timer_frames = 0
        self._pending_one_timer_passer_control = None
        self._pending_one_timer_target = None
        self._pass_button_release_frames = 0
        self._shot_button_release_frames = 0
        self._trace_path = os.environ.get('CLASSIC_AI_TRACE_PATH')
        self._trace_frame = 0
        self._last_decision = 'init'

    def predict(self, state, deterministic=True):
        action = np.zeros((1, GameConsts.INPUT_MAX), dtype=np.int8)
        self._last_action_preferences = action.astype(np.float32)
        return action, None

    def predict_game_state(self, game_state, deterministic=True):
        action = self._predict_actions(game_state, deterministic)
        return np.asarray([action], dtype=np.int8)

    def get_action_preferences(self, _state=None):
        return self._last_action_preferences

    def learn(self, *args, **kwargs):
        raise NotImplementedError("ClassicAI is inference-only. Use scripts/play.py to run it.")

    def save(self, *args, **kwargs):
        raise NotImplementedError("ClassicAI does not produce a trainable checkpoint.")

    def _predict_actions(self, game_state, deterministic):
        t1 = game_state.team1
        t2 = game_state.team2
        if self._pending_one_timer_frames > 0:
            self._pending_one_timer_frames -= 1
            if self._pending_one_timer_frames == 0:
                self._pending_one_timer_passer_control = None
                self._pending_one_timer_target = None

        controlled = t1.get_controlled_player()
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        self._trace_frame += 1

        if t1.goalie_haspuck:
            self._reset_possession_tracking()
            self._clear_one_timer_window()
            return self._finalize(self._predict_goalie_possession(game_state, deterministic))

        if t1.player_haspuck:
            if t1.control == 0:
                self._reset_possession_tracking()
                self._clear_one_timer_window()
                action[GameConsts.INPUT_B] = 1
                return self._finalize(action)
            if self._should_finish_one_timer(controlled, game_state.engine, t1.control):
                return self._finalize(self._finish_one_timer(controlled, t2.goalie, game_state.engine))
            return self._finalize(self._predict_offense(game_state, controlled, deterministic))

        one_timer_receiver = self._find_pending_one_timer_receiver(t1)
        if one_timer_receiver is not None and not (t2.player_haspuck or t2.goalie_haspuck):
            # The engine can flag the receiver as one-timer-ready while the pass is
            # still travelling. Do not spend the shot input or clear the armed
            # window until team possession/control actually switches to the receiver.
            return self._finalize(action)

        self._reset_possession_tracking()
        if t2.player_haspuck or t2.goalie_haspuck:
            self._clear_one_timer_window()
        if t2.player_haspuck or t2.goalie_haspuck:
            return self._finalize(self._predict_defense(game_state, controlled))

        return self._finalize(self._predict_loose_puck(game_state, controlled))

    def _predict_goalie_possession(self, game_state, deterministic):
        t1 = game_state.team1
        goalie = t1.goalie
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        target = self._select_goalie_pass_target(t1, goalie, deterministic)
        if target is not None:
            target_player = target[0]
            self._steer_toward(action, goalie, target_player.x, target_player.y, deadzone=4)
            self._try_tap_pass(action)
            return action

        self._steer_toward(action, goalie, 0, -185, deadzone=4)
        return action

    def _predict_offense(self, game_state, controlled, deterministic):
        t1 = game_state.team1
        t2 = game_state.team2
        engine = game_state.engine
        opp_goalie = t2.goalie
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        is_full_rink = len(t1.players) >= 5
        nearest_opp = self._nearest_opponent_distance(controlled, t2.players)
        corner_trapped = self._is_corner_trapped(controlled, t2.players)
        behind_net = self._is_behind_opponent_net(controlled)
        side_setup = self._is_side_setup(controlled)
        stall_frames = self._update_possession_tracking(t1, controlled)
        attack_plan = self._current_attack_plan

        pass_target = self._select_pass_target(
            t1,
            controlled,
            deterministic,
            corner_trapped=corner_trapped,
            is_full_rink=is_full_rink,
            stall_frames=stall_frames,
            attack_plan=attack_plan,
        )
        progression_pass = self._select_progression_pass_target(
            t1,
            controlled,
            t2.players,
            is_full_rink=is_full_rink,
            stall_frames=stall_frames,
            attack_plan=attack_plan,
        )
        shoot_score = self._score_shot(controlled, opp_goalie, engine)
        scoring_pass = self._select_scoring_pass_target(t1, controlled, opp_goalie, attack_plan=attack_plan)
        carry_options = self._build_attack_targets(
            controlled,
            opp_goalie,
            corner_trapped=corner_trapped,
            is_full_rink=is_full_rink,
            stall_frames=stall_frames,
            attack_plan=attack_plan,
        )
        carry_target = self._choose_weighted_target(carry_options, deterministic)

        support_pass_ready = self._is_support_pass_ready(controlled, pass_target)
        scoring_pass_ready = self._is_scoring_pass_ready(controlled, scoring_pass)
        safe_pass_ready = bool(pass_target is not None and getattr(pass_target[0], 'passing_lane_clear', False))
        play_pass = self._select_playmaking_pass_target(
            t1,
            controlled,
            t2.players,
            attack_plan=attack_plan,
            stall_frames=stall_frames,
            is_full_rink=is_full_rink,
        )
        forced_one_timer_pass = self._select_one_timer_pass_target(
            t1,
            controlled,
            opp_goalie,
            is_full_rink=is_full_rink,
            attack_plan=attack_plan,
        )
        forced_one_timer_ready = self._should_force_one_timer_attempt(
            controlled,
            forced_one_timer_pass,
            shoot_score,
            attack_plan,
            stall_frames,
            nearest_opp,
        )
        one_timer_pass_ready = self._is_one_timer_pass_ready(
            controlled,
            scoring_pass,
            shoot_score,
            attack_plan,
            side_setup,
            is_full_rink,
        )

        if behind_net:
            if forced_one_timer_ready:
                target_player = forced_one_timer_pass[0]
                self._steer_toward(action, controlled, target_player.x, target_player.y)
                self._try_tap_pass(action, target_player, passer_control=t1.control, force_one_timer=True)
                self._last_decision = 'behind_net_one_timer_feed'
                return action

            if scoring_pass_ready and (scoring_pass[1] >= 1.0 or one_timer_pass_ready):
                target_player = scoring_pass[0]
                self._steer_toward(action, controlled, target_player.x, target_player.y)
                self._try_tap_pass(action, target_player, passer_control=t1.control, force_one_timer=True)
                self._last_decision = 'behind_net_scoring_feed'
                return action

            exit_target = self._build_behind_net_exit_target(controlled, t2.players)
            self._steer_toward(action, controlled, exit_target[0], exit_target[1])
            self._last_decision = 'behind_net_exit'
            return action

        if nearest_opp < 22 and not scoring_pass_ready and not forced_one_timer_ready:
            pressure_pass = progression_pass if progression_pass is not None and progression_pass[1] >= 0.70 else None
            if pressure_pass is None and safe_pass_ready and pass_target[1] >= 0.95 and self._pass_gains_space(controlled, pass_target[0]):
                pressure_pass = pass_target

            if pressure_pass is not None:
                target_player = pressure_pass[0]
                self._steer_toward(action, controlled, target_player.x, target_player.y)
                self._try_tap_pass(action, target_player, passer_control=t1.control)
                self._last_decision = 'pressure_outlet_pass'
                return action

            protect_target = self._build_pressure_escape_target(controlled, t2.players)
            self._steer_toward(action, controlled, protect_target[0], protect_target[1])
            self._last_decision = 'pressure_escape'
            return action

        if self._should_make_progression_pass(controlled, progression_pass, stall_frames, nearest_opp, shoot_score):
            target_player = progression_pass[0]
            self._steer_toward(action, controlled, target_player.x, target_player.y)
            self._try_tap_pass(action, target_player, passer_control=t1.control)
            self._last_decision = 'progression_pass'
            return action

        if self._should_make_play_pass(controlled, play_pass, attack_plan, stall_frames, nearest_opp, shoot_score):
            target_player = play_pass[0]
            self._steer_toward(action, controlled, target_player.x, target_player.y)
            self._try_tap_pass(action, target_player, passer_control=t1.control, force_one_timer=False)
            self._last_decision = 'playmaking_pass'
            return action

        if is_full_rink and stall_frames >= 10:
            if safe_pass_ready and pass_target[1] >= 0.85:
                target_player = pass_target[0]
                self._steer_toward(action, controlled, target_player.x, target_player.y)
                self._try_tap_pass(action, target_player, passer_control=t1.control)
                self._last_decision = 'stall_reset_pass'
                return action

            reset_target = self._build_reset_target(controlled)
            self._steer_toward(action, controlled, reset_target[0], reset_target[1])
            self._last_decision = 'stall_reset_carry'
            return action

        if forced_one_timer_ready:
            target_player = forced_one_timer_pass[0]
            self._steer_toward(action, controlled, target_player.x, target_player.y)
            self._try_tap_pass(action, target_player, passer_control=t1.control, force_one_timer=True)
            self._last_decision = 'forced_one_timer_feed'
            return action

        if one_timer_pass_ready or (scoring_pass_ready and scoring_pass[1] >= max(1.20, shoot_score - 0.10)):
            target_player = scoring_pass[0]
            self._steer_toward(action, controlled, target_player.x, target_player.y)
            self._try_tap_pass(action, target_player, passer_control=t1.control, force_one_timer=True)
            self._last_decision = 'scoring_pass'
            return action

        if is_full_rink and support_pass_ready and pass_target[1] >= 1.0 and controlled.y < 195:
            target_player = pass_target[0]
            self._steer_toward(action, controlled, target_player.x, target_player.y)
            self._try_tap_pass(action, target_player, passer_control=t1.control)
            self._last_decision = 'support_pass'
            return action

        if corner_trapped and safe_pass_ready and pass_target[1] >= 1.0:
            target_player = pass_target[0]
            self._steer_toward(action, controlled, target_player.x, target_player.y)
            self._try_tap_pass(action, target_player, passer_control=t1.control)
            self._last_decision = 'corner_escape_pass'
            return action

        if is_full_rink and safe_pass_ready and pass_target[1] >= 1.35 and controlled.y < 185:
            target_player = pass_target[0]
            self._steer_toward(action, controlled, target_player.x, target_player.y)
            self._try_tap_pass(action, target_player, passer_control=t1.control)
            self._last_decision = 'zone_entry_pass'
            return action

        if safe_pass_ready and pass_target[1] >= max(1.6, shoot_score + 0.2):
            target_player = pass_target[0]
            self._steer_toward(action, controlled, target_player.x, target_player.y)
            self._try_tap_pass(action, target_player, passer_control=t1.control)
            self._last_decision = 'high_value_pass'
            return action

        if shoot_score >= 1.75:
            self._shoot(action, controlled, opp_goalie, engine)
            self._last_decision = 'shoot_high_score'
            return action

        if carry_target is not None:
            self._steer_toward(action, controlled, carry_target[0], carry_target[1])
            self._last_decision = 'carry_attack_target'

        if side_setup and controlled.y >= 198 and scoring_pass_ready and scoring_pass[1] >= 1.20:
            self._try_tap_pass(action, scoring_pass[0], passer_control=t1.control, force_one_timer=True)
            self._last_decision = 'side_setup_feed'

        if safe_pass_ready and pass_target[1] >= 1.7 and (nearest_opp < 32 or corner_trapped):
            self._try_tap_pass(action, pass_target[0], passer_control=t1.control)
            self._last_decision = 'late_safety_pass'

        if is_full_rink and safe_pass_ready and pass_target[1] >= 1.15 and controlled.y < 170 and abs(controlled.x) > 38:
            self._try_tap_pass(action, pass_target[0], passer_control=t1.control)
            self._last_decision = 'wide_entry_pass'

        if attack_plan == 1 and support_pass_ready and pass_target[1] >= 0.95:
            self._try_tap_pass(action, pass_target[0], passer_control=t1.control)
            self._last_decision = 'planned_support_pass'

        if shoot_score >= 1.35 and controlled.y >= 178:
            self._shoot(action, controlled, opp_goalie, engine)
            self._last_decision = 'shoot_medium_score'

        return action

    def _predict_defense(self, game_state, controlled):
        t1 = game_state.team1
        t2 = game_state.team2
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)

        if t2.player_haspuck:
            carrier = t2.get_controlled_player()
        elif t2.goalie_haspuck:
            carrier = t2.goalie
        else:
            carrier = game_state.puck

        target_x, target_y = self._build_defensive_target(carrier, controlled)
        self._steer_toward(action, controlled, target_x, target_y)

        if self._distance_xy(controlled.x, controlled.y, carrier.x, carrier.y) < 18:
            action[GameConsts.INPUT_B] = 1

        if controlled.y < -214 and carrier.y > -224:
            self._steer_toward(action, controlled, 0, -176)

        return action

    def _predict_loose_puck(self, game_state, controlled):
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        puck = game_state.puck
        if puck.y < -214 and controlled.y > -196:
            target_x = int(puck.x * 0.45)
            target_y = -188
        else:
            target_x = puck.x + puck.vx * 2
            target_y = puck.y + puck.vy * 2
        target_y = max(target_y, -218)
        self._steer_toward(action, controlled, target_x, target_y)
        return action

    def _select_pass_target(self, team, controlled, deterministic, corner_trapped=False, is_full_rink=False, stall_frames=0, attack_plan=0):
        candidates = []
        controlled_idx = team.control - 1 if team.control > 0 else -1
        for index, player in enumerate(team.players):
            if index == controlled_idx:
                continue

            score = 0.0
            if player.passing_lane_clear:
                score += 1.2
            else:
                score -= 0.9
            if player.y > controlled.y + 8:
                score += 0.9
            elif corner_trapped and player.y <= controlled.y - 18:
                score += 0.8
            elif is_full_rink and player.y > controlled.y - 10:
                score += 0.35
            if abs(player.x) < 28:
                score += 0.5
            if is_full_rink and abs(player.x) < abs(controlled.x):
                score += 0.45
            if corner_trapped and abs(player.x) + 10 < abs(controlled.x):
                score += 0.8
            if player.is_one_timer:
                score += 0.9
            score += min(max((player.y - controlled.y) / 120.0, 0.0), 0.8)
            if corner_trapped:
                score += min(max((controlled.y - player.y) / 140.0, 0.0), 0.5)
            if is_full_rink and controlled.y < 165:
                if player.passing_lane_clear and player.y > controlled.y + 12:
                    score += 0.55
                if abs(player.x) <= 24:
                    score += 0.35
                if abs(player.x) > abs(controlled.x) + 18:
                    score -= 0.35
            if is_full_rink and abs(controlled.x) > 44 and abs(player.x) + 12 < abs(controlled.x):
                score += 0.55
            if attack_plan == 1:
                if player.passing_lane_clear:
                    score += 0.50
                if player.y >= controlled.y - 8:
                    score += 0.30
                if abs(player.x) < abs(controlled.x):
                    score += 0.35
            elif attack_plan == 2:
                if abs(player.x) <= 24:
                    score += 0.40
                if player.y > controlled.y + 10:
                    score += 0.25
            elif attack_plan == 3:
                if controlled.x * player.x < 0:
                    score += 0.45
                if player.passing_lane_clear:
                    score += 0.20
            if stall_frames >= 6:
                if player.passing_lane_clear:
                    score += 0.45
                if abs(player.x) + 8 < abs(controlled.x):
                    score += 0.45
                if player.y < controlled.y - 12:
                    score += 0.30
            if stall_frames >= 10 and abs(player.x) <= 28:
                score += 0.55
            score -= self._distance_xy(controlled.x, controlled.y, player.x, player.y) / 250.0
            candidates.append((player, score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        best_score = candidates[0][1]
        top = [item for item in candidates if item[1] >= best_score - 0.25]
        if deterministic:
            return self._choose_pass_candidate(top, controlled, attack_plan)
        if len(top) == 1:
            return top[0]
        return self._rng.choice(top)

    def _score_shot(self, controlled, goalie, engine):
        score = 0.0
        side_setup = self._is_side_setup(controlled)
        if controlled.y >= 180:
            score += 0.8
        if controlled.y >= 198:
            score += 0.35
        if GameConsts.CREASE_LOWER_BOUND <= controlled.y <= GameConsts.CREASE_UPPER_BOUND:
            score += 0.8
        if abs(controlled.x) <= GameConsts.CREASE_MAX_X + 8:
            score += 0.4
        if abs(controlled.x) <= GameConsts.CREASE_MAX_X + 18 and controlled.y >= 195:
            score += 0.45
        if side_setup:
            score += 0.45
        if abs(goalie.x - controlled.x) >= 10:
            score += 0.6
        if abs(goalie.x) >= 12 and controlled.y >= 190:
            score += 0.35
        if goalie.is_pad_stack or goalie.is_dive:
            score += 0.9
        if engine.goalie_box_small:
            score += 0.3
        if controlled.is_breakaway or engine.breakaway_context:
            score += 0.7
        if abs(controlled.vx) >= GameConsts.CREASE_MIN_VEL:
            score += 0.3
        if engine.one_timer_collision_mode:
            score += 0.35
        if engine.in_close_top_shelf:
            score += 0.25
        return score

    def _score_receiver_chance(self, receiver, goalie):
        score = 0.0
        if receiver.y >= 190:
            score += 0.7
        if receiver.y >= GameConsts.CREASE_LOWER_BOUND:
            score += 0.45
        if abs(receiver.x) <= GameConsts.CREASE_MAX_X + 12:
            score += 0.8
        if abs(goalie.x - receiver.x) >= 8:
            score += 0.35
        if receiver.is_one_timer:
            score += 0.85
        return score

    def _select_playmaking_pass_target(self, team, controlled, opponents, attack_plan=0, stall_frames=0, is_full_rink=False):
        if len(team.players) < 2:
            return None

        candidates = []
        controlled_idx = team.control - 1 if team.control > 0 else -1
        zone = self._rink_zone(controlled.y)
        for index, player in enumerate(team.players):
            if index == controlled_idx:
                continue

            distance = self._distance_xy(controlled.x, controlled.y, player.x, player.y)
            if distance < 16 or distance > 190:
                continue

            score = 0.0
            if getattr(player, 'passing_lane_clear', False):
                score += 1.0
            else:
                score -= 0.35

            y_gain = player.y - controlled.y
            toward_middle = abs(player.x) < abs(controlled.x)
            receiver_pressure = self._nearest_opponent_distance(player, opponents)

            if zone == 'defensive':
                if y_gain > 10:
                    score += min(y_gain / 55.0, 1.4)
                if player.y > -80:
                    score += 0.65
                if abs(player.x) <= 48:
                    score += 0.35
                if player.y < controlled.y - 8:
                    score -= 1.0
            elif zone == 'neutral':
                if y_gain > 8:
                    score += min(y_gain / 70.0, 1.0)
                if -15 <= player.y <= 145:
                    score += 0.45
                if toward_middle:
                    score += 0.45
                if controlled.x * player.x < 0 and abs(controlled.x) > 28:
                    score += 0.35
            elif zone == 'entry':
                if player.y >= controlled.y - 16:
                    score += 0.45
                if player.y >= 170:
                    score += 0.35
                if abs(player.x) <= 58:
                    score += 0.65
                if controlled.x * player.x < 0:
                    score += 0.45
            else:
                if abs(player.x) <= 64 and player.y >= 164:
                    score += 0.85
                if controlled.x * player.x < 0:
                    score += 0.55
                if player.y <= controlled.y + 20:
                    score += 0.30
                if player.y < 145:
                    score -= 0.55

            if receiver_pressure > 24:
                score += min(receiver_pressure / 65.0, 0.65)
            else:
                score -= 0.45
            if is_full_rink:
                score += 0.20
            if attack_plan in (1, 3):
                score += 0.20
            if stall_frames >= 4:
                score += 0.35
            score -= distance / 360.0
            candidates.append((player, score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        if candidates[0][1] < 0.65:
            return None
        return candidates[0]

    def _select_progression_pass_target(self, team, controlled, opponents, is_full_rink=False, stall_frames=0, attack_plan=0):
        if len(team.players) < 2:
            return None

        candidates = []
        controlled_idx = team.control - 1 if team.control > 0 else -1
        zone = self._rink_zone(controlled.y)
        for index, player in enumerate(team.players):
            if index == controlled_idx:
                continue

            distance = self._distance_xy(controlled.x, controlled.y, player.x, player.y)
            if distance < 18 or distance > 205:
                continue

            y_gain = player.y - controlled.y
            if zone in ('defensive', 'neutral') and y_gain < 6:
                continue
            if zone == 'entry' and y_gain < -10 and stall_frames < 5:
                continue
            if zone == 'attack' and player.y < 158:
                continue

            score = 0.0
            lane_clear = getattr(player, 'passing_lane_clear', False)
            if lane_clear:
                score += 1.45
            else:
                score -= 1.15

            score += min(max(y_gain / 62.0, -0.8), 1.75)
            if controlled.y < 0x58 <= player.y:
                score += 1.10
            elif controlled.y < 120 and player.y >= 120:
                score += 0.75
            if controlled.y < 170 and player.y >= 170:
                score += 0.65
            if 0x58 <= player.y <= GameConsts.P2_NET_Y and abs(player.x) <= 0x47:
                score += 0.70
            if abs(player.x) <= 35:
                score += 0.30
            if abs(player.x) + 8 < abs(controlled.x):
                score += 0.35

            receiver_pressure = self._nearest_opponent_distance(player, opponents)
            if receiver_pressure >= 34:
                score += 0.70
            elif receiver_pressure >= 24:
                score += 0.35
            else:
                score -= 0.75

            if 28 <= distance <= 150:
                score += 0.35
            elif distance > 170:
                score -= 0.35
            if is_full_rink:
                score += 0.20
            if attack_plan == 1 and lane_clear:
                score += 0.30
            elif attack_plan == 2 and abs(player.x) <= 0x47:
                score += 0.30
            elif attack_plan == 3 and controlled.x * player.x < 0:
                score += 0.25
            if stall_frames >= 4:
                score += 0.25
            if stall_frames >= 8 and y_gain >= -8:
                score += 0.30

            score -= distance / 420.0
            candidates.append((player, score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        if candidates[0][1] < 0.55:
            return None
        top_score = candidates[0][1]
        top = [item for item in candidates if item[1] >= top_score - 0.25]
        return self._choose_progression_candidate(top, controlled, attack_plan)

    def _should_make_progression_pass(self, controlled, progression_pass, stall_frames, nearest_opp, shoot_score):
        if progression_pass is None:
            return False

        receiver, pass_score = progression_pass
        zone = self._rink_zone(controlled.y)
        y_gain = receiver.y - controlled.y
        if zone == 'defensive':
            return pass_score >= 0.65 and y_gain >= 10
        if zone == 'neutral':
            return pass_score >= 0.75 and (y_gain >= 14 or receiver.y >= 0x58)
        if zone == 'entry':
            return pass_score >= 0.95 and shoot_score < 1.65
        if zone == 'attack':
            if nearest_opp < 30 and pass_score >= 1.05:
                return True
            if stall_frames >= 5 and pass_score >= 0.95:
                return True
            return y_gain >= 8 and pass_score >= max(1.25, shoot_score - 0.35)
        return False

    def _should_make_play_pass(self, controlled, play_pass, attack_plan, stall_frames, nearest_opp, shoot_score):
        if play_pass is None:
            return False

        receiver, pass_score = play_pass
        zone = self._rink_zone(controlled.y)
        if zone == 'defensive':
            return pass_score >= 0.75 and receiver.y > controlled.y + 8
        if nearest_opp < 20 and pass_score >= 0.80:
            return True
        if stall_frames >= 5 and pass_score >= 0.70:
            return True
        if zone == 'neutral':
            return pass_score >= 0.95 and receiver.y >= controlled.y - 6
        if zone == 'entry':
            return pass_score >= 1.05 and shoot_score < 1.55
        if zone == 'attack':
            if attack_plan in (1, 3) and pass_score >= 0.85:
                return True
            return pass_score >= max(1.15, shoot_score - 0.60)
        return False

    def _rink_zone(self, y):
        if y < -80:
            return 'defensive'
        if y < 105:
            return 'neutral'
        if y < 178:
            return 'entry'
        return 'attack'

    def _select_one_timer_pass_target(self, team, controlled, goalie, is_full_rink=False, attack_plan=0):
        if len(team.players) < 2 or controlled.y < 120:
            return None

        candidates = []
        controlled_idx = team.control - 1 if team.control > 0 else -1
        min_y = 140 if controlled.y < 175 else max(140, controlled.y - 86)
        for index, player in enumerate(team.players):
            if index == controlled_idx:
                continue
            if player.y < min_y or player.y > GameConsts.P2_NET_Y - 20:
                continue
            if abs(player.x) > 98:
                continue

            distance = self._distance_xy(controlled.x, controlled.y, player.x, player.y)
            if distance < 14 or distance > 185:
                continue

            score = 0.0
            if getattr(player, 'passing_lane_clear', False):
                score += 0.80
            else:
                score -= 0.05
            if getattr(player, 'is_one_timer', 0.0):
                score += 3.0
            if controlled.x * player.x < 0:
                score += 1.05
            if abs(player.x) <= 58:
                score += 0.90
            if player.y < 150 and not getattr(player, 'is_one_timer', 0.0):
                score -= 0.80
            score += min(max((player.y - 140) / 90.0, 0.0), 0.70)
            if player.y >= 170:
                score += 0.35
            if player.y >= 188:
                score += 0.65
            if controlled.y >= 170 and player.y <= controlled.y + 18:
                score += 0.65
            if abs(goalie.x - player.x) >= 8:
                score += 0.25
            if is_full_rink:
                score += 0.25
            if attack_plan in (1, 3):
                score += 0.45
            if 24 <= distance <= 125:
                score += 0.35
            score -= distance / 310.0
            candidates.append((player, score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        if candidates[0][1] < 0.35:
            return None
        return candidates[0]

    def _select_scoring_pass_target(self, team, controlled, goalie, attack_plan=0):
        candidates = []
        controlled_idx = team.control - 1 if team.control > 0 else -1
        for index, player in enumerate(team.players):
            if index == controlled_idx:
                continue

            score = self._score_receiver_chance(player, goalie)
            if player.passing_lane_clear:
                score += 0.95
            else:
                score -= 1.25
            if controlled.x * player.x < 0:
                score += 0.75
            if abs(player.x) + 6 < abs(controlled.x):
                score += 0.55
            if player.y >= controlled.y - 6:
                score += 0.35
            if attack_plan == 1 and player.passing_lane_clear:
                score += 0.30
            if attack_plan == 3 and controlled.x * player.x < 0:
                score += 0.35
            score -= self._distance_xy(controlled.x, controlled.y, player.x, player.y) / 280.0
            candidates.append((player, score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        top_score = candidates[0][1]
        top = [item for item in candidates if item[1] >= top_score - 0.2]
        return self._choose_pass_candidate(top, controlled, attack_plan)

    def _select_goalie_pass_target(self, team, goalie, deterministic):
        candidates = []
        for player in team.players:
            score = 0.0
            if player.passing_lane_clear:
                score += 1.0
            else:
                score -= 0.25
            if player.y > -20:
                score += 0.8
            elif player.y > -115:
                score += 0.35
            if abs(player.x) < 42:
                score += 0.5
            opp_dist = getattr(player, 'dist_to_controlled_opp', 0) or 0
            if opp_dist > 20:
                score += min(opp_dist / 40.0, 0.8)
            if player.y > goalie.y + 45:
                score += 0.4
            score -= self._distance_xy(goalie.x, goalie.y, player.x, player.y) / 320.0
            candidates.append((player, score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        best_score = candidates[0][1]
        top = [item for item in candidates if item[1] >= best_score - 0.2]
        if deterministic:
            return max(top, key=lambda item: (item[0].y, -abs(item[0].x), item[1]))
        return self._rng.choice(top)

    def _build_attack_targets(self, controlled, goalie, corner_trapped=False, is_full_rink=False, stall_frames=0, attack_plan=0):
        if self._attack_lane_frames <= 0:
            self._attack_lane_bias = self._rng.choice((-1, 1))
            self._attack_lane_frames = self._rng.randint(20, 45)
        else:
            self._attack_lane_frames -= 1

        favored_side = self._attack_lane_bias
        if attack_plan == 2:
            favored_side = 0
        elif attack_plan == 3:
            favored_side = -1 if controlled.x >= 0 else 1
        options = []
        if corner_trapped:
            escape_side = -1 if controlled.x > 0 else 1
            targets = [
                ((0, 196), 1.45),
                ((escape_side * 24, 188), 1.35),
                ((int(controlled.x * 0.45), 178), 1.20),
                ((0, 170), 1.05),
            ]
            for target, score in targets:
                score -= self._distance_xy(controlled.x, controlled.y, target[0], target[1]) / 250.0
                options.append((target, score))
            return options

        if is_full_rink and stall_frames >= 6:
            reset_targets = [
                ((0, min(196, controlled.y + 10)), 1.55),
                ((int(controlled.x * 0.35), max(150, controlled.y - 20)), 1.35),
                ((0, max(140, controlled.y - 28)), 1.20),
            ]
            for target, score in reset_targets:
                score -= self._distance_xy(controlled.x, controlled.y, target[0], target[1]) / 245.0
                if abs(target[0]) < abs(controlled.x):
                    score += 0.25
                options.append((target, score))
            return options

        if controlled.y >= 165:
            attack_side = 1 if controlled.x >= 0 else -1
            scoring_targets = [
                ((attack_side * 46, 214), 1.60),
                ((0, 216), 1.48),
                ((-attack_side * 18, 222), 1.42),
            ]
            if attack_plan == 2:
                scoring_targets.insert(0, ((0, 208), 1.72))
            elif attack_plan == 3:
                scoring_targets.insert(0, ((-attack_side * 34, 218), 1.68))
            for target, score in scoring_targets:
                score -= self._distance_xy(controlled.x, controlled.y, target[0], target[1]) / 255.0
                if abs(target[0]) < abs(controlled.x):
                    score += 0.12
                if abs(goalie.x - target[0]) >= 10:
                    score += 0.10
                options.append((target, score))
            return options

        if is_full_rink:
            if controlled.y < 40:
                stage_targets = [
                    ((0, 88), 1.45),
                    ((favored_side * 18, 100), 1.30),
                    ((-favored_side * 16, 92), 1.20),
                ]
            elif controlled.y < 120:
                stage_targets = [
                    ((0, 148), 1.50),
                    ((favored_side * 24, 156), 1.34),
                    ((int(controlled.x * 0.4), 152), 1.24),
                ]
            elif controlled.y < 185:
                stage_targets = [
                    ((0, 188), 1.48),
                    ((favored_side * 24, 196), 1.32),
                    ((-favored_side * 14, 186), 1.22),
                ]
            else:
                stage_targets = []

            for target, score in stage_targets:
                score -= self._distance_xy(controlled.x, controlled.y, target[0], target[1]) / 255.0
                if abs(target[0]) < abs(controlled.x):
                    score += 0.18
                options.append((target, score))

            if options:
                return options

        for side in (favored_side, -favored_side, 0):
            if side == 0:
                target = (0, 205)
            else:
                target = (side * 42, 210)
            score = 1.0
            if side != 0:
                score += abs(goalie.x - target[0]) / 35.0
            if side == -1 and goalie.x > 4:
                score += 0.25
            if side == 1 and goalie.x < -4:
                score += 0.25
            if side == favored_side:
                score += 0.2
            score -= self._distance_xy(controlled.x, controlled.y, target[0], target[1]) / 260.0
            options.append((target, score))
        return options

    def _is_corner_trapped(self, controlled, opponents: Sequence):
        if controlled.y < 198 or abs(controlled.x) < 52:
            return False
        return self._nearest_opponent_distance(controlled, opponents) < 40

    def _is_behind_opponent_net(self, controlled):
        return controlled.y >= (GameConsts.P2_NET_Y - 24)

    def _is_side_setup(self, controlled):
        return controlled.y >= (GameConsts.CREASE_LOWER_BOUND - 10) and abs(controlled.x) > (GameConsts.CREASE_MAX_X + 6) and abs(controlled.x) < 78

    def _build_behind_net_exit_target(self, controlled, opponents: Sequence):
        pressure_side = 0
        if opponents:
            nearest = min(opponents, key=lambda opponent: self._distance_xy(controlled.x, controlled.y, opponent.x, opponent.y))
            pressure_side = 1 if nearest.x >= controlled.x else -1

        if pressure_side == 0:
            pressure_side = 1 if controlled.x >= 0 else -1

        exit_side = -pressure_side
        exit_x = exit_side * max(20, min(42, abs(controlled.x) + 8))
        exit_y = 222 if abs(controlled.x) > 12 else 214
        return (exit_x, exit_y)

    def _build_reset_target(self, controlled):
        target_x = 0 if abs(controlled.x) > 28 else int(controlled.x * 0.4)
        target_y = max(148, min(192, controlled.y - 20))
        return (target_x, target_y)

    def _progress_score(self, controlled):
        return float(controlled.y - (abs(controlled.x) * 0.75))

    def _update_possession_tracking(self, team, controlled):
        control_key = team.control
        progress_score = self._progress_score(controlled)

        if self._last_possession_control != control_key:
            self._possession_frames = 1
            self._stall_frames = 0
            self._advance_attack_plan()
        else:
            self._possession_frames += 1
            if self._last_progress_score is None or progress_score > self._last_progress_score + 4.0:
                self._stall_frames = max(0, self._stall_frames - 2)
            else:
                self._stall_frames += 1

        self._last_possession_control = control_key
        self._last_progress_score = progress_score
        return self._stall_frames

    def _reset_possession_tracking(self):
        self._last_possession_control = None
        self._last_progress_score = None
        self._possession_frames = 0
        self._stall_frames = 0

    def _advance_attack_plan(self):
        self._attack_plan_index = (self._attack_plan_index + 1) % 4
        self._current_attack_plan = self._attack_plan_index

    def _build_defensive_target(self, carrier, controlled):
        if carrier.y < -224:
            target_x = int(carrier.x * 0.35)
            target_y = -194
        elif carrier.y < -150:
            target_x = int(carrier.x * 0.55)
            target_y = max(-188, carrier.y + 18)
        elif carrier.y < -80:
            target_x = int(carrier.x * 0.65)
            target_y = carrier.y + 10
        else:
            target_x = carrier.x + carrier.vx * 2
            target_y = carrier.y + carrier.vy * 2

        if controlled.y < -205 and carrier.y > -224:
            target_x = 0
            target_y = -172

        return (int(np.clip(target_x, -75, 75)), int(np.clip(target_y, -208, 220)))

    def _arm_one_timer_window(self, receiver, passer_control=None, force=False):
        dangerous_receiver = (
            getattr(receiver, 'y', 0) >= 178
            and abs(getattr(receiver, 'x', 0)) <= 78
        )
        if force or (getattr(receiver, 'passing_lane_clear', False) and (getattr(receiver, 'is_one_timer', 0.0) or dangerous_receiver)):
            self._pending_one_timer_frames = 42
            self._pending_one_timer_passer_control = passer_control
            self._pending_one_timer_target = (getattr(receiver, 'x', 0), getattr(receiver, 'y', 0))

    def _clear_one_timer_window(self):
        self._pending_one_timer_frames = 0
        self._pending_one_timer_passer_control = None
        self._pending_one_timer_target = None

    def _should_finish_one_timer(self, controlled, engine, current_control=None):
        if self._pending_one_timer_frames <= 0:
            return False
        if (
            current_control is not None
            and self._pending_one_timer_passer_control is not None
            and current_control == self._pending_one_timer_passer_control
            and not getattr(controlled, 'is_one_timer', 0.0)
            and not getattr(engine, 'one_timer_collision_mode', 0.0)
        ):
            return False
        if getattr(controlled, 'is_one_timer', 0.0) or getattr(engine, 'one_timer_collision_mode', 0.0):
            return True
        if controlled.y < 178:
            return False
        if abs(controlled.x) <= 78:
            return True
        if self._pending_one_timer_target is None:
            return False
        target_x, target_y = self._pending_one_timer_target
        return self._distance_xy(controlled.x, controlled.y, target_x, target_y) <= 45 and controlled.y >= 175

    def _find_pending_one_timer_receiver(self, team):
        if self._pending_one_timer_frames <= 0:
            return None

        candidates = []
        for player in team.players:
            if getattr(player, 'is_one_timer', 0.0):
                candidates.append((player, 3.0))
                continue
            if self._pending_one_timer_target is None:
                continue
            target_x, target_y = self._pending_one_timer_target
            if player.y >= 175 and self._distance_xy(player.x, player.y, target_x, target_y) <= 34:
                candidates.append((player, 1.0))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]

    def _should_force_one_timer_attempt(self, controlled, one_timer_pass, shoot_score, attack_plan, stall_frames, nearest_opp):
        if one_timer_pass is None:
            return False

        receiver, pass_score = one_timer_pass
        if controlled.y < 120:
            return False
        if getattr(receiver, 'is_one_timer', 0.0):
            return True
        if controlled.y >= 150 and pass_score >= 0.35:
            return True
        if controlled.y >= 196 and pass_score >= 0.20:
            return True
        if nearest_opp < 28 and pass_score >= 0.35:
            return True
        if attack_plan in (1, 3) and pass_score >= 0.30:
            return True
        if stall_frames >= 3 and pass_score >= 0.30:
            return True
        return pass_score >= max(0.45, shoot_score - 2.50)

    def _finish_one_timer(self, controlled, goalie, engine):
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        if self._try_tap_shot(action):
            self._clear_one_timer_window()
        return action

    def _try_tap_shot(self, action):
        if self._shot_button_release_frames > 0:
            action[GameConsts.INPUT_C] = 0
            return False

        action[GameConsts.INPUT_C] = 1
        return True

    def _try_tap_pass(self, action, receiver=None, passer_control=None, force_one_timer=False):
        if self._pass_button_release_frames > 0:
            action[GameConsts.INPUT_B] = 0
            return False

        if receiver is not None and force_one_timer:
            self._arm_one_timer_window(receiver, passer_control=passer_control, force=True)
        action[GameConsts.INPUT_B] = 1
        return True

    def _build_pressure_escape_target(self, controlled, opponents: Sequence):
        if not opponents:
            return self._build_reset_target(controlled)

        nearest = min(opponents, key=lambda opponent: self._distance_xy(controlled.x, controlled.y, opponent.x, opponent.y))
        defender_ahead = nearest.y >= controlled.y - 4
        if abs(controlled.x - nearest.x) < 8:
            lateral_dir = -1 if controlled.x >= 0 else 1
        else:
            lateral_dir = 1 if controlled.x > nearest.x else -1

        if abs(controlled.x) > 52:
            escape_x = int(controlled.x * 0.45)
        else:
            lane_width = 32 if defender_ahead else 22
            escape_x = controlled.x + lateral_dir * lane_width

        if defender_ahead:
            escape_y = controlled.y + (6 if controlled.y < 170 else -8)
        else:
            escape_y = controlled.y + 24

        if controlled.y < 110:
            escape_y = max(escape_y, controlled.y + 16)

        escape_x = int(np.clip(escape_x, -62, 62))
        escape_y = int(np.clip(escape_y, max(-190, controlled.y - 12), 220))
        return (escape_x, escape_y)

    def _pass_gains_space(self, controlled, receiver):
        return bool(
            receiver.y >= controlled.y - 6
            or receiver.y >= 0x58
            or abs(receiver.x) + 12 < abs(controlled.x)
        )

    def _choose_progression_candidate(self, candidates, controlled, attack_plan):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        return max(
            candidates,
            key=lambda item: (
                int(item[0].passing_lane_clear),
                int(controlled.y < 0x58 <= item[0].y),
                min(item[0].y - controlled.y, 90),
                int(abs(item[0].x) <= 0x47),
                int(attack_plan == 3 and controlled.x * item[0].x < 0),
                item[1],
            ),
        )

    def _choose_pass_candidate(self, candidates, controlled, attack_plan):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if attack_plan == 1:
            return max(
                candidates,
                key=lambda item: (
                    int(item[0].passing_lane_clear),
                    item[0].y,
                    -abs(item[0].x),
                    item[1],
                ),
            )
        if attack_plan == 2:
            return max(
                candidates,
                key=lambda item: (
                    -abs(item[0].x),
                    item[0].y,
                    int(item[0].passing_lane_clear),
                    item[1],
                ),
            )
        if attack_plan == 3:
            return max(
                candidates,
                key=lambda item: (
                    int(controlled.x * item[0].x < 0),
                    int(item[0].passing_lane_clear),
                    item[0].y,
                    item[1],
                ),
            )
        return candidates[0]

    def _is_support_pass_ready(self, controlled, pass_target):
        if pass_target is None:
            return False
        receiver = pass_target[0]
        return bool(
            receiver.passing_lane_clear
            and receiver.y >= controlled.y - 12
            and abs(receiver.x) <= abs(controlled.x) + 12
        )

    def _is_scoring_pass_ready(self, controlled, scoring_pass):
        if scoring_pass is None:
            return False
        receiver = scoring_pass[0]
        min_receiver_y = controlled.y - (32 if controlled.y >= 196 else 10)
        return bool(
            receiver.passing_lane_clear
            and receiver.y >= min_receiver_y
            and (controlled.x * receiver.x < 0 or abs(receiver.x) <= GameConsts.CREASE_MAX_X + 18)
        )

    def _is_one_timer_pass_ready(self, controlled, scoring_pass, shoot_score, attack_plan, side_setup, is_full_rink):
        if not self._is_scoring_pass_ready(controlled, scoring_pass):
            return False

        receiver, pass_score = scoring_pass
        dangerous_receiver = receiver.y >= 178 and abs(receiver.x) <= 78
        if not dangerous_receiver:
            return False
        if pass_score < 1.35:
            return False

        high_value_lane = pass_score >= shoot_score - 0.90
        planned_pass_tactic = attack_plan in (1, 3)
        deep_setup = controlled.y >= 196 or side_setup
        full_rink_slot_feed = is_full_rink and controlled.y >= 176 and receiver.y >= controlled.y - 16
        return high_value_lane or planned_pass_tactic or deep_setup or full_rink_slot_feed

    def _choose_weighted_target(self, options, deterministic):
        if not options:
            return None

        options = sorted(options, key=lambda item: item[1], reverse=True)
        best_score = options[0][1]
        top = [item for item in options if item[1] >= best_score - 0.2]
        if deterministic or len(top) == 1:
            return top[0][0]

        weights = [max(item[1], 0.05) for item in top]
        return self._rng.choices([item[0] for item in top], weights=weights, k=1)[0]

    def _shoot(self, action, controlled, goalie, engine):
        if self._is_side_setup(controlled):
            target_x = int(np.clip(-goalie.x * 1.4, -18, 18))
            target_y = min(GameConsts.P2_NET_Y, controlled.y + 12)
        else:
            side = 1 if controlled.x >= goalie.x else -1
            target_x = controlled.x + side * 20
            target_y = min(GameConsts.P2_NET_Y, controlled.y + 18)

        self._steer_toward(action, controlled, target_x, target_y)
        if goalie.is_pad_stack or goalie.is_dive or engine.in_close_top_shelf or self._is_side_setup(controlled):
            action[GameConsts.INPUT_MODE] = 1
        action[GameConsts.INPUT_C] = 1

    def _steer_toward(self, action, controlled, target_x, target_y, deadzone=6):
        delta_x = target_x - controlled.x
        delta_y = target_y - controlled.y

        if delta_x <= -deadzone:
            action[GameConsts.INPUT_LEFT] = 1
            action[GameConsts.INPUT_RIGHT] = 0
        elif delta_x >= deadzone:
            action[GameConsts.INPUT_RIGHT] = 1
            action[GameConsts.INPUT_LEFT] = 0

        if delta_y <= -deadzone:
            action[GameConsts.INPUT_UP] = 1
            action[GameConsts.INPUT_DOWN] = 0
        elif delta_y >= deadzone:
            action[GameConsts.INPUT_DOWN] = 1
            action[GameConsts.INPUT_UP] = 0

    def _nearest_opponent_distance(self, controlled, opponents: Sequence):
        if not opponents:
            return 999.0
        return min(self._distance_xy(controlled.x, controlled.y, opponent.x, opponent.y) for opponent in opponents)

    def _distance_xy(self, x1, y1, x2, y2):
        return float(np.hypot(x1 - x2, y1 - y2))

    def _trace_action(self, action):
        if not self._trace_path:
            return

        needs_header = not os.path.exists(self._trace_path)
        with open(self._trace_path, 'a', encoding='utf-8') as trace_file:
            if needs_header:
                trace_file.write('frame,decision,b,c,mode,up,down,left,right,pending_one_timer,target_x,target_y,pass_release,shot_release\n')
            target_x = ''
            target_y = ''
            if self._pending_one_timer_target is not None:
                target_x, target_y = self._pending_one_timer_target
            trace_file.write(
                f"{self._trace_frame},{self._last_decision},"
                f"{int(action[GameConsts.INPUT_B])},{int(action[GameConsts.INPUT_C])},{int(action[GameConsts.INPUT_MODE])},"
                f"{int(action[GameConsts.INPUT_UP])},{int(action[GameConsts.INPUT_DOWN])},"
                f"{int(action[GameConsts.INPUT_LEFT])},{int(action[GameConsts.INPUT_RIGHT])},"
                f"{self._pending_one_timer_frames},{target_x},{target_y},"
                f"{self._pass_button_release_frames},{self._shot_button_release_frames}\n"
            )

    def _finalize(self, action):
        if action[GameConsts.INPUT_B]:
            if self._pass_button_release_frames > 0:
                action[GameConsts.INPUT_B] = 0
                self._pass_button_release_frames -= 1
            else:
                self._pass_button_release_frames = 3
        elif self._pass_button_release_frames > 0:
            self._pass_button_release_frames -= 1

        if action[GameConsts.INPUT_C] and not action[GameConsts.INPUT_MODE]:
            if self._shot_button_release_frames > 0:
                action[GameConsts.INPUT_C] = 0
                self._shot_button_release_frames -= 1
            else:
                self._shot_button_release_frames = 1
        elif not action[GameConsts.INPUT_MODE] and self._shot_button_release_frames > 0:
            self._shot_button_release_frames -= 1

        self._trace_action(action)
        prefs = action.astype(np.float32)
        self._last_action_preferences = prefs.reshape(1, -1)
        return action