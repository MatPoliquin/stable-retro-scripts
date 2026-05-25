import math
import os

import numpy as np

from game_wrappers.nhl94.nhl94_const import GameConsts


class ClassicAIModel:
    """Focused scripted controller for catching the puck.

    This version intentionally does one thing: skate the controlled player to
    the best reachable puck intercept point, using puck/player velocities and a
    conservative poke/check input when the puck is within reach.
    """

    MIN_X = -112
    MAX_X = 112
    MIN_Y = -246
    MAX_Y = 246

    MAX_LOOKAHEAD_FRAMES = 36
    MIN_LOOKAHEAD_FRAMES = 1
    PLAYER_SPEED_ESTIMATE = 7.2
    BURST_DISTANCE = 22.0
    POKE_DISTANCE = 17.0
    POKE_AHEAD_DISTANCE = 24.0
    POKE_COOLDOWN_FRAMES = 9
    SWITCH_COOLDOWN_FRAMES = 14
    SWITCH_SCORE_ADVANTAGE = 18.0
    ATTACK_ENTRY_Y = 130
    ATTACK_CIRCLE_CENTER_Y = 198
    ATTACK_CIRCLE_RADIUS_X = 54
    ATTACK_CIRCLE_RADIUS_Y = 30
    PRESSURE_DISTANCE = 32.0
    GOALIE_PASS_LEAD_FRAMES = 8
    GOALIE_AVOID_DISTANCE = 34.0
    NET_FRONT_PROGRESS = 206
    BEHIND_NET_PROGRESS = 238
    GOALIE_PASS_RETRY_FRAMES = 12
    ONE_TIMER_PASS_RETRY_FRAMES = 18
    ONE_TIMER_SHOT_DELAY_FRAMES = 1
    ONE_TIMER_SHOT_FRAMES = 16
    ONE_TIMER_PASS_LEAD_FRAMES = 7
    CARRY_MIN_SPACE = 34.0
    CARRY_LANE_MIN_SPACE = 26.0

    def __init__(self, args=None, env=None):
        self.args = args
        self.env = env
        self._last_action_preferences = np.zeros((1, GameConsts.INPUT_MAX), dtype=np.float32)
        self._trace_path = os.environ.get("CLASSIC_AI_TRACE_PATH")
        self._trace_frame = 0
        self._last_decision = "init"
        self._last_target = (0, 0)
        self._poke_cooldown = 0
        self._switch_cooldown = 0
        self._goalie_pass_cooldown = 0
        self._one_timer_pass_cooldown = 0
        self._one_timer_shot_delay = 0
        self._one_timer_shot_frames = 0
        self._circle_phase = 0.0
        self._circle_direction = 1

    def predict(self, state, deterministic=True):
        action = np.zeros((1, GameConsts.INPUT_MAX), dtype=np.int8)
        self._last_action_preferences = action.astype(np.float32)
        return action, None

    def predict_game_state(self, game_state, deterministic=True):
        action = self._predict_actions(game_state)
        return np.asarray([action], dtype=np.int8)

    def get_action_preferences(self, _state=None):
        return self._last_action_preferences

    def learn(self, *args, **kwargs):
        raise NotImplementedError("ClassicAI is inference-only. Use scripts/play.py to run it.")

    def save(self, *args, **kwargs):
        raise NotImplementedError("ClassicAI does not produce a trainable checkpoint.")

    def _predict_actions(self, game_state):
        self._trace_frame += 1
        if self._poke_cooldown > 0:
            self._poke_cooldown -= 1
        if self._switch_cooldown > 0:
            self._switch_cooldown -= 1
        if self._goalie_pass_cooldown > 0:
            self._goalie_pass_cooldown -= 1
        if self._one_timer_pass_cooldown > 0:
            self._one_timer_pass_cooldown -= 1

        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        team = game_state.team1
        opponents = game_state.team2
        controlled = team.get_controlled_player()
        puck = game_state.puck

        if team.player_haspuck:
            action = self._predict_puck_control(controlled, team, opponents)
            return self._finalize(action)

        if team.goalie_haspuck:
            action = self._predict_goalie_pass(team, opponents)
            return self._finalize(action)

        self._goalie_pass_cooldown = 0

        if self._one_timer_shot_frames > 0 and not (opponents.player_haspuck or opponents.goalie_haspuck):
            if self._one_timer_shot_delay > 0:
                self._one_timer_shot_delay -= 1
                self._last_decision = "one_timer_wait_for_pass"
            else:
                action[GameConsts.INPUT_C] = 1
                self._one_timer_shot_frames -= 1
                self._last_decision = "one_timer_shoot"
            return self._finalize(action)

        if opponents.player_haspuck or opponents.goalie_haspuck:
            self._one_timer_shot_delay = 0
            self._one_timer_shot_frames = 0

        if self._should_switch_player(team, controlled, puck, opponents):
            action[GameConsts.INPUT_A] = 1
            self._switch_cooldown = self.SWITCH_COOLDOWN_FRAMES
            self._last_decision = "switch_to_interceptor"
            return self._finalize(action)

        target = self._choose_intercept_target(controlled, puck, opponents)
        self._last_target = target
        self._steer_toward(action, controlled, target[0], target[1])

        distance_to_target = self._distance_xy(self._x(controlled), self._y(controlled), target[0], target[1])
        distance_to_puck = self._distance_between(controlled, puck)
        closing_speed = self._closing_speed(controlled, puck)

        if self._should_burst(distance_to_target, distance_to_puck, closing_speed):
            action[GameConsts.INPUT_C] = 1

        if self._should_poke(controlled, puck, opponents):
            action[GameConsts.INPUT_B] = 1
            self._poke_cooldown = self.POKE_COOLDOWN_FRAMES
            self._last_decision = "intercept_poke"
        elif opponents.player_haspuck or opponents.goalie_haspuck:
            self._last_decision = "intercept_carrier"
        else:
            self._last_decision = "intercept_loose_puck"

        return self._finalize(action)

    def _predict_puck_control(self, controlled, team, opponents):
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        one_timer_target = self._choose_one_timer_target(controlled, team, opponents)
        if one_timer_target is not None:
            return self._predict_one_timer_pass(action, controlled, one_timer_target)

        target = self._choose_puck_control_target(controlled, opponents)
        self._last_target = target
        self._steer_toward(action, controlled, target[0], target[1], deadzone=5)
        return action

    def _predict_one_timer_pass(self, action, controlled, target):
        self._last_target = target
        self._steer_toward(action, controlled, target[0], target[1], deadzone=3)
        if self._one_timer_pass_cooldown <= 0:
            action[GameConsts.INPUT_B] = 1
            self._one_timer_pass_cooldown = self.ONE_TIMER_PASS_RETRY_FRAMES
            self._one_timer_shot_delay = self.ONE_TIMER_SHOT_DELAY_FRAMES
            self._one_timer_shot_frames = self.ONE_TIMER_SHOT_FRAMES
            self._last_decision = "one_timer_pass_tap"
        else:
            self._last_decision = "one_timer_pass_release"
        return action

    def _predict_goalie_pass(self, team, opponents):
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        goalie = team.goalie
        target = self._choose_goalie_pass_target(goalie, team.players, opponents.players)
        self._last_target = target
        self._steer_toward(action, goalie, target[0], target[1], deadzone=3)
        if self._goalie_pass_cooldown <= 0:
            action[GameConsts.INPUT_B] = 1
            self._goalie_pass_cooldown = self.GOALIE_PASS_RETRY_FRAMES
            self._last_decision = "goalie_pass_teammate_tap"
        else:
            self._last_decision = "goalie_pass_teammate_release"
        return action

    def _choose_goalie_pass_target(self, goalie, teammates, opponents):
        if not teammates:
            return (0, self.ATTACK_ENTRY_Y)

        goalie_y = self._y(goalie)
        attack_sign = 1 if goalie_y <= 0 else -1
        best_player = teammates[0]
        best_score = -float("inf")

        for player in teammates:
            target_x = self._x(player) + self._vx(player) * self.GOALIE_PASS_LEAD_FRAMES * 0.20
            target_y = self._y(player) + self._vy(player) * self.GOALIE_PASS_LEAD_FRAMES * 0.20
            distance_from_goalie = self._distance_xy(self._x(goalie), goalie_y, target_x, target_y)
            forward_progress = (target_y - goalie_y) * attack_sign
            opponent_space = self._nearest_point_distance(target_x, target_y, opponents)
            lane_space = self._pass_lane_space(goalie, target_x, target_y, opponents)

            score = opponent_space * 1.6 + lane_space * 1.2 + forward_progress * 0.65
            score -= distance_from_goalie * 0.22
            if getattr(player, "passing_lane_clear", False):
                score += 18.0
            if forward_progress < 8:
                score -= 35.0
            if target_y * attack_sign < -110:
                score -= 20.0

            if score > best_score:
                best_score = score
                best_player = player

        lead_x = self._x(best_player) + self._vx(best_player) * self.GOALIE_PASS_LEAD_FRAMES * 0.20
        lead_y = self._y(best_player) + self._vy(best_player) * self.GOALIE_PASS_LEAD_FRAMES * 0.20
        return (
            int(np.clip(lead_x, -96, 96)),
            int(np.clip(lead_y, self.MIN_Y + 28, self.MAX_Y - 28)),
        )

    def _choose_one_timer_target(self, controlled, team, opponents):
        attack_sign = self._attack_sign(opponents)
        passer_progress = self._progress(self._y(controlled), attack_sign)
        if passer_progress < self.ATTACK_ENTRY_Y + 10:
            return None
        if self._one_timer_pass_cooldown > 0:
            return None

        best_target = None
        best_score = 54.0
        controlled_idx = team.control - 1 if getattr(team, "control", 0) > 0 else -1
        opponent_group = self._players_and_goalie(opponents)

        for index, player in enumerate(team.players):
            if index == controlled_idx:
                continue
            if getattr(player, "is_falling", 0.0) or getattr(player, "is_dive", 0.0):
                continue

            target_x = self._x(player) + self._vx(player) * self.ONE_TIMER_PASS_LEAD_FRAMES * 0.20
            target_y = self._y(player) + self._vy(player) * self.ONE_TIMER_PASS_LEAD_FRAMES * 0.20
            target_progress = self._progress(target_y, attack_sign)
            if target_progress < self.ATTACK_ENTRY_Y + 12 or target_progress > self.NET_FRONT_PROGRESS + 16:
                continue
            if abs(target_x) > 88:
                continue

            lane_space = self._pass_lane_space(controlled, target_x, target_y, opponent_group)
            receiver_space = self._nearest_point_distance(target_x, target_y, opponent_group)
            if lane_space < 22.0 or receiver_space < 24.0:
                continue

            lateral_separation = abs(target_x - self._x(controlled))
            vertical_separation = abs(target_progress - passer_progress)
            if lateral_separation < 28 and vertical_separation < 18:
                continue

            goalie = getattr(opponents, "goalie", None)
            goalie_gap = self._distance_xy(target_x, target_y, self._x(goalie), self._y(goalie)) if goalie is not None else 999.0
            if goalie_gap < self.GOALIE_AVOID_DISTANCE:
                continue

            slot_bonus = max(0.0, 64.0 - abs(target_x)) * 0.25
            one_timer_depth = 28.0 - abs(target_progress - 184.0) * 0.25
            score = lane_space * 1.4 + receiver_space * 1.2 + lateral_separation * 0.35 + slot_bonus + one_timer_depth
            score -= self._net_danger_penalty(target_x, target_y, opponents, attack_sign, allow_behind_net=False) * 0.35
            if getattr(player, "passing_lane_clear", False):
                score += 16.0
            if target_progress < passer_progress - 28:
                score -= 18.0

            if score > best_score:
                best_score = score
                best_target = (int(np.clip(target_x, -96, 96)), int(np.clip(target_y, self.MIN_Y, self.MAX_Y)))

        return best_target

    def _choose_puck_control_target(self, controlled, opponents):
        attack_sign = self._attack_sign(opponents)
        opponent_group = self._players_and_goalie(opponents)
        nearest = self._nearest_opponent(controlled, opponent_group)
        if nearest is not None and self._distance_between(controlled, nearest) < self.PRESSURE_DISTANCE:
            self._last_decision = "carry_escape_pressure"
            return self._pressure_escape_target(controlled, nearest, opponents, attack_sign)

        if self._progress(self._y(controlled), attack_sign) < self.ATTACK_ENTRY_Y:
            self._last_decision = "carry_to_attack_zone"
            return self._attack_entry_target(controlled, opponents, attack_sign)

        self._last_decision = "carry_attack_circle"
        return self._attack_circle_target(controlled, opponents, attack_sign)

    def _attack_entry_target(self, controlled, opponents, attack_sign):
        player_progress = self._progress(self._y(controlled), attack_sign)
        candidate_progress = int(np.clip(max(self.ATTACK_ENTRY_Y + 18, player_progress + 64), self.ATTACK_ENTRY_Y, 188))
        lanes = (-72, -42, -18, 18, 42, 72)
        candidates = [(lane_x, self._signed_y(candidate_progress, attack_sign)) for lane_x in lanes]
        return self._best_empty_target(controlled, candidates, opponents, prefer_forward=True, attack_sign=attack_sign)

    def _attack_circle_target(self, controlled, opponents, attack_sign):
        current_target = self._circle_point(self._circle_phase, attack_sign)
        if self._distance_xy(self._x(controlled), self._y(controlled), current_target[0], current_target[1]) < 16:
            self._circle_phase += self._circle_direction * 0.75

        phase_options = [self._circle_phase + self._circle_direction * step for step in (0.55, 1.05, 1.55, 2.15)]
        candidates = [self._circle_point(phase, attack_sign) for phase in phase_options]
        candidates.extend(self._open_cycle_candidates(controlled, attack_sign))
        allow_behind_net = self._should_use_behind_net(controlled, opponents, attack_sign)
        if allow_behind_net:
            candidates.extend(self._behind_net_candidates(controlled, attack_sign))
        target = self._best_empty_target(
            controlled,
            candidates,
            opponents,
            prefer_forward=False,
            attack_sign=attack_sign,
            allow_behind_net=allow_behind_net,
        )

        if target == candidates[-1]:
            self._circle_phase += self._circle_direction * 0.35
        else:
            self._circle_phase += self._circle_direction * 0.08

        if abs(self._x(controlled)) > 88 or self._progress(self._y(controlled), attack_sign) > 235:
            self._circle_direction *= -1
        return target

    def _circle_point(self, phase, attack_sign):
        return (
            int(np.clip(math.cos(phase) * self.ATTACK_CIRCLE_RADIUS_X, -84, 84)),
            self._signed_y(int(np.clip(self.ATTACK_CIRCLE_CENTER_Y + math.sin(phase) * self.ATTACK_CIRCLE_RADIUS_Y, 150, 226)), attack_sign),
        )

    def _pressure_escape_target(self, controlled, nearest, opponents, attack_sign):
        player_x = self._x(controlled)
        player_y = self._y(controlled)
        player_progress = self._progress(player_y, attack_sign)
        away_x = player_x - self._x(nearest)
        away_progress = player_progress - self._progress(self._y(nearest), attack_sign)
        length = max(1.0, math.hypot(away_x, away_progress))

        base_x = player_x + away_x / length * 44.0
        base_progress = player_progress + away_progress / length * 30.0
        if base_progress < self.ATTACK_ENTRY_Y and player_progress >= self.ATTACK_ENTRY_Y:
            base_progress = self.ATTACK_ENTRY_Y + 12
        elif player_progress < self.ATTACK_ENTRY_Y:
            base_progress = max(base_progress, player_progress + 28)

        candidates = [
            (base_x, self._signed_y(base_progress, attack_sign)),
            (-base_x * 0.55, self._signed_y(max(base_progress, self.ATTACK_ENTRY_Y + 22), attack_sign)),
            (0, self._signed_y(max(player_progress + 18, self.ATTACK_ENTRY_Y + 18), attack_sign)),
            (math.copysign(72, player_x if player_x else away_x), self._signed_y(max(self.ATTACK_ENTRY_Y + 12, player_progress - 10), attack_sign)),
        ]
        allow_behind_net = player_progress >= self.ATTACK_ENTRY_Y and self._should_use_behind_net(controlled, opponents, attack_sign)
        if allow_behind_net:
            candidates.extend(self._behind_net_candidates(controlled, attack_sign))
        clamped = [(int(np.clip(x_pos, -86, 86)), int(np.clip(y_pos, self.MIN_Y, self.MAX_Y))) for x_pos, y_pos in candidates]
        return self._best_empty_target(
            controlled,
            clamped,
            opponents,
            prefer_forward=player_progress < self.ATTACK_ENTRY_Y,
            attack_sign=attack_sign,
            allow_behind_net=allow_behind_net,
        )

    def _best_empty_target(self, controlled, candidates, opponents, prefer_forward, attack_sign=1, allow_behind_net=False):
        best_target = candidates[0]
        best_score = -float("inf")
        player_x = self._x(controlled)
        player_y = self._y(controlled)
        player_progress = self._progress(player_y, attack_sign)
        opponent_group = self._players_and_goalie(opponents) if hasattr(opponents, "goalie") else opponents

        for target_x, target_y in candidates:
            target_progress = self._progress(target_y, attack_sign)
            nearest_space = self._nearest_point_distance(target_x, target_y, opponent_group)
            lane_space = self._pass_lane_space(controlled, target_x, target_y, opponent_group)
            travel_distance = self._distance_xy(player_x, player_y, target_x, target_y)
            score = min(nearest_space, 70.0) * 1.35 + min(lane_space, 60.0) * 0.65 - travel_distance * 0.18
            if prefer_forward:
                score += max(0, target_progress - player_progress) * 0.35
            else:
                score -= max(0, target_progress - 232) * 2.0
                score -= max(0, 146 - target_progress) * 0.7
                score -= max(0, abs(target_x) - 82) * 0.7
            if nearest_space < self.CARRY_MIN_SPACE:
                score -= (self.CARRY_MIN_SPACE - nearest_space) * 5.0
            if lane_space < self.CARRY_LANE_MIN_SPACE:
                score -= (self.CARRY_LANE_MIN_SPACE - lane_space) * 4.0
            score -= self._net_danger_penalty(target_x, target_y, opponents, attack_sign, allow_behind_net)
            if score > best_score:
                best_score = score
                best_target = (int(target_x), int(target_y))
        return best_target

    def _open_cycle_candidates(self, controlled, attack_sign):
        player_x = self._x(controlled)
        player_progress = self._progress(self._y(controlled), attack_sign)
        side = math.copysign(1.0, player_x if abs(player_x) > 8 else self._circle_direction)
        raw_candidates = [
            (side * 82, 176),
            (-side * 82, 176),
            (side * 76, 208),
            (-side * 76, 204),
            (side * 54, 160),
            (-side * 54, 164),
            (0, 154),
        ]
        if player_progress > self.NET_FRONT_PROGRESS - 8 and abs(player_x) < 48:
            raw_candidates.extend([(side * 84, 190), (-side * 84, 190)])
        return [
            (int(np.clip(x_pos, -88, 88)), self._signed_y(int(np.clip(progress, 146, 226)), attack_sign))
            for x_pos, progress in raw_candidates
        ]

    def _attack_sign(self, opponents):
        opponent_net_y = self._y(getattr(opponents, "net", None))
        if opponent_net_y != 0:
            return 1 if opponent_net_y > 0 else -1
        return 1 if self._y(getattr(opponents, "goalie", None)) >= 0 else -1

    def _progress(self, y_pos, attack_sign):
        return y_pos * attack_sign

    def _signed_y(self, progress, attack_sign):
        return int(np.clip(progress * attack_sign, self.MIN_Y, self.MAX_Y))

    def _players_and_goalie(self, team):
        players = list(getattr(team, "players", []) or [])
        goalie = getattr(team, "goalie", None)
        if goalie is not None:
            players.append(goalie)
        return players

    def _should_use_behind_net(self, controlled, opponents, attack_sign):
        player_x = self._x(controlled)
        player_progress = self._progress(self._y(controlled), attack_sign)
        if player_progress < 180:
            return False

        goalie = getattr(opponents, "goalie", None)
        goalie_distance = self._distance_between(controlled, goalie) if goalie is not None else 999.0
        skater_pressure = self._nearest_point_distance(self._x(controlled), self._y(controlled), getattr(opponents, "players", []))
        central_net_lane = abs(player_x) < 38 and player_progress >= self.NET_FRONT_PROGRESS - 8
        return goalie_distance < 42.0 or skater_pressure < self.PRESSURE_DISTANCE or central_net_lane

    def _behind_net_candidates(self, controlled, attack_sign):
        player_x = self._x(controlled)
        side = math.copysign(1.0, player_x if abs(player_x) > 6 else self._circle_direction)
        progress_values = (self.BEHIND_NET_PROGRESS, self.BEHIND_NET_PROGRESS - 8)
        side_values = (side * 78, -side * 78, side * 52)
        return [
            (int(np.clip(x_pos, -88, 88)), self._signed_y(progress, attack_sign))
            for progress in progress_values
            for x_pos in side_values
        ]

    def _net_danger_penalty(self, target_x, target_y, opponents, attack_sign, allow_behind_net):
        target_progress = self._progress(target_y, attack_sign)
        penalty = 0.0

        goalie = getattr(opponents, "goalie", None)
        if goalie is not None:
            goalie_distance = self._distance_xy(target_x, target_y, self._x(goalie), self._y(goalie))
            if goalie_distance < self.GOALIE_AVOID_DISTANCE:
                penalty += (self.GOALIE_AVOID_DISTANCE - goalie_distance) * 5.0

        if target_progress >= self.NET_FRONT_PROGRESS and abs(target_x) < 58:
            central_width = max(0.0, 42.0 - abs(target_x))
            crease_depth = max(0.0, target_progress - self.NET_FRONT_PROGRESS)
            depth_weight = (58.0 - abs(target_x)) / 58.0
            penalty += central_width * 2.4 + crease_depth * 1.4 * depth_weight

        if target_progress >= 228 and abs(target_x) < 46:
            penalty += (46.0 - abs(target_x)) * 4.5

        if target_progress >= self.BEHIND_NET_PROGRESS - 4:
            if abs(target_x) < 46:
                penalty += (46.0 - abs(target_x)) * 6.0
            elif allow_behind_net:
                penalty -= 24.0
            else:
                penalty += 35.0

        return penalty

    def _should_switch_player(self, team, controlled, puck, opponents):
        if self._switch_cooldown > 0 or len(team.players) <= 1:
            return False
        if self._distance_between(controlled, puck) <= self.POKE_DISTANCE + 4:
            return False

        current_score = self._intercept_score(controlled, puck, opponents)
        best_score = current_score
        best_player = controlled

        for player in team.players:
            score = self._intercept_score(player, puck, opponents)
            if score < best_score:
                best_score = score
                best_player = player

        if best_player is controlled:
            return False
        return best_score + self.SWITCH_SCORE_ADVANTAGE < current_score

    def _intercept_score(self, player, puck, opponents):
        player_x = self._x(player)
        player_y = self._y(player)
        player_vx = self._vx(player)
        player_vy = self._vy(player)
        puck_x = self._x(puck)
        puck_y = self._y(puck)
        puck_vx = self._vx(puck)
        puck_vy = self._vy(puck)

        best_score = float("inf")
        for frames in range(self.MIN_LOOKAHEAD_FRAMES, self.MAX_LOOKAHEAD_FRAMES + 1):
            target_x, target_y = self._predict_puck_position(puck_x, puck_y, puck_vx, puck_vy, frames)
            projected_x = player_x + player_vx * min(frames, 8) * 0.20
            projected_y = player_y + player_vy * min(frames, 8) * 0.20
            distance = self._distance_xy(projected_x, projected_y, target_x, target_y)
            reachable_distance = self.PLAYER_SPEED_ESTIMATE * frames
            arrival_error = max(0.0, distance - reachable_distance)
            score = arrival_error + frames * 0.45
            if score < best_score:
                best_score = score

        carrier = self._opponent_puck_carrier(opponents)
        if carrier is not None:
            best_score += self._distance_between(player, carrier) * 0.25
        return best_score

    def _choose_intercept_target(self, controlled, puck, opponents):
        player_x = self._x(controlled)
        player_y = self._y(controlled)
        player_vx = self._vx(controlled)
        player_vy = self._vy(controlled)
        puck_x = self._x(puck)
        puck_y = self._y(puck)
        puck_vx = self._vx(puck)
        puck_vy = self._vy(puck)

        best_target = (puck_x, puck_y)
        best_error = float("inf")

        for frames in range(self.MIN_LOOKAHEAD_FRAMES, self.MAX_LOOKAHEAD_FRAMES + 1):
            predicted_puck_x, predicted_puck_y = self._predict_puck_position(puck_x, puck_y, puck_vx, puck_vy, frames)
            desired_player_x = player_x + player_vx * min(frames, 8) * 0.20
            desired_player_y = player_y + player_vy * min(frames, 8) * 0.20
            distance = self._distance_xy(desired_player_x, desired_player_y, predicted_puck_x, predicted_puck_y)
            reachable_distance = self.PLAYER_SPEED_ESTIMATE * frames
            error = abs(distance - reachable_distance) + frames * 0.15

            if error < best_error:
                best_error = error
                best_target = (predicted_puck_x, predicted_puck_y)

        carrier = self._opponent_puck_carrier(opponents)
        if carrier is not None:
            carrier_target = self._carrier_stick_target(carrier, puck)
            carrier_weight = 0.65 if self._distance_between(controlled, carrier) < 42 else 0.35
            best_target = (
                best_target[0] * (1.0 - carrier_weight) + carrier_target[0] * carrier_weight,
                best_target[1] * (1.0 - carrier_weight) + carrier_target[1] * carrier_weight,
            )

        return (
            int(np.clip(best_target[0], self.MIN_X, self.MAX_X)),
            int(np.clip(best_target[1], self.MIN_Y, self.MAX_Y)),
        )

    def _predict_puck_position(self, puck_x, puck_y, puck_vx, puck_vy, frames):
        predicted_x = puck_x + puck_vx * frames * 0.45
        predicted_y = puck_y + puck_vy * frames * 0.45

        if predicted_x < self.MIN_X:
            predicted_x = self.MIN_X + (self.MIN_X - predicted_x) * 0.65
        elif predicted_x > self.MAX_X:
            predicted_x = self.MAX_X - (predicted_x - self.MAX_X) * 0.65

        if predicted_y < self.MIN_Y:
            predicted_y = self.MIN_Y + (self.MIN_Y - predicted_y) * 0.45
        elif predicted_y > self.MAX_Y:
            predicted_y = self.MAX_Y - (predicted_y - self.MAX_Y) * 0.45

        return predicted_x, predicted_y

    def _carrier_stick_target(self, carrier, puck):
        carrier_x = self._x(carrier)
        carrier_y = self._y(carrier)
        puck_x = self._x(puck)
        puck_y = self._y(puck)
        carrier_vx = self._vx(carrier)
        carrier_vy = self._vy(carrier)

        toward_puck_x = puck_x - carrier_x
        toward_puck_y = puck_y - carrier_y
        length = max(1.0, math.hypot(toward_puck_x, toward_puck_y))
        stick_x = carrier_x + toward_puck_x / length * 10.0
        stick_y = carrier_y + toward_puck_y / length * 10.0
        return (stick_x + carrier_vx * 0.65, stick_y + carrier_vy * 0.65)

    def _should_burst(self, distance_to_target, distance_to_puck, closing_speed):
        if distance_to_puck <= 10:
            return False
        if distance_to_target > self.BURST_DISTANCE:
            return True
        return closing_speed < 1.5 and distance_to_puck > 14

    def _should_poke(self, controlled, puck, opponents):
        if self._poke_cooldown > 0:
            return False

        distance_to_puck = self._distance_between(controlled, puck)
        if distance_to_puck > self.POKE_DISTANCE:
            return False

        puck_ahead = self._is_point_in_front(controlled, self._x(puck), self._y(puck), self.POKE_AHEAD_DISTANCE)
        if puck_ahead:
            return True

        carrier = self._opponent_puck_carrier(opponents)
        if carrier is None:
            return distance_to_puck <= 10

        return self._distance_between(controlled, carrier) <= 18

    def _opponent_puck_carrier(self, opponents):
        if opponents.player_haspuck or opponents.goalie_haspuck:
            return opponents.get_controlled_player()
        return None

    def _is_point_in_front(self, player, x_pos, y_pos, max_forward_distance):
        orientation = int(getattr(player, "orientation", 0) or 0)
        forward_x, forward_y = self._orientation_vector(orientation)
        delta_x = x_pos - self._x(player)
        delta_y = y_pos - self._y(player)
        forward_distance = delta_x * forward_x + delta_y * forward_y
        lateral_distance = abs(delta_x * -forward_y + delta_y * forward_x)
        return 0 <= forward_distance <= max_forward_distance and lateral_distance <= 13

    def _orientation_vector(self, orientation):
        angle = orientation * (2.0 * math.pi / 8.0)
        return math.cos(angle), math.sin(angle)

    def _closing_speed(self, controlled, puck):
        delta_x = self._x(puck) - self._x(controlled)
        delta_y = self._y(puck) - self._y(controlled)
        distance = max(1.0, math.hypot(delta_x, delta_y))
        relative_vx = self._vx(controlled) - self._vx(puck)
        relative_vy = self._vy(controlled) - self._vy(puck)
        return (relative_vx * delta_x + relative_vy * delta_y) / distance

    def _steer_toward(self, action, player, target_x, target_y, deadzone=4):
        delta_x = target_x - self._x(player)
        delta_y = target_y - self._y(player)

        if delta_x <= -deadzone:
            action[GameConsts.INPUT_LEFT] = 1
        elif delta_x >= deadzone:
            action[GameConsts.INPUT_RIGHT] = 1

        if delta_y <= -deadzone:
            action[GameConsts.INPUT_DOWN] = 1
        elif delta_y >= deadzone:
            action[GameConsts.INPUT_UP] = 1

    def _x(self, obj):
        return int(getattr(obj, "x", 0) or 0)

    def _y(self, obj):
        return int(getattr(obj, "y", 0) or 0)

    def _vx(self, obj):
        return int(getattr(obj, "vx", 0) or 0)

    def _vy(self, obj):
        return int(getattr(obj, "vy", 0) or 0)

    def _distance_between(self, first, second):
        return self._distance_xy(self._x(first), self._y(first), self._x(second), self._y(second))

    def _distance_xy(self, x1, y1, x2, y2):
        return float(math.hypot(x1 - x2, y1 - y2))

    def _nearest_opponent(self, player, opponents):
        if not opponents:
            return None
        return min(opponents, key=lambda opponent: self._distance_between(player, opponent))

    def _nearest_point_distance(self, x_pos, y_pos, players):
        if not players:
            return 999.0
        return min(self._distance_xy(x_pos, y_pos, self._x(player), self._y(player)) for player in players)

    def _pass_lane_space(self, passer, target_x, target_y, opponents):
        if not opponents:
            return 999.0

        start_x = self._x(passer)
        start_y = self._y(passer)
        lane_x = target_x - start_x
        lane_y = target_y - start_y
        lane_length_sq = max(1.0, lane_x * lane_x + lane_y * lane_y)
        best_space = 999.0

        for opponent in opponents:
            opponent_x = self._x(opponent)
            opponent_y = self._y(opponent)
            projection = ((opponent_x - start_x) * lane_x + (opponent_y - start_y) * lane_y) / lane_length_sq
            projection = min(1.0, max(0.0, projection))
            closest_x = start_x + lane_x * projection
            closest_y = start_y + lane_y * projection
            best_space = min(best_space, self._distance_xy(opponent_x, opponent_y, closest_x, closest_y))

        return best_space

    def _trace_action(self, action):
        if not self._trace_path:
            return

        needs_header = not os.path.exists(self._trace_path)
        with open(self._trace_path, "a", encoding="utf-8") as trace_file:
            if needs_header:
                trace_file.write("frame,decision,target_x,target_y,a,b,c,up,down,left,right,poke_cooldown,switch_cooldown\n")
            trace_file.write(
                f"{self._trace_frame},{self._last_decision},{self._last_target[0]},{self._last_target[1]},"
                f"{int(action[GameConsts.INPUT_A])},{int(action[GameConsts.INPUT_B])},{int(action[GameConsts.INPUT_C])},"
                f"{int(action[GameConsts.INPUT_UP])},{int(action[GameConsts.INPUT_DOWN])},"
                f"{int(action[GameConsts.INPUT_LEFT])},{int(action[GameConsts.INPUT_RIGHT])},"
                f"{self._poke_cooldown},{self._switch_cooldown}\n"
            )

    def _finalize(self, action):
        self._trace_action(action)
        preferences = action.astype(np.float32)
        self._last_action_preferences = preferences.reshape(1, -1)
        return action