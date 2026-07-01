"""Mechanics-aware NHL94 ClassicAI v2.

This scripted oracle is intentionally separate from ``classic_ai.py``. It uses
the NHL94 disassembly deep-dive as the design target: evaluate high-level
hockey options from puck/goalie/collision geometry, then emit either low-level
Genesis buttons or the wrapper's HOCKEY_INTENT_DPAD action format.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_intents import (
    HOCKEY_INTENT_DPAD_ACTION_SPACE,
    HOCKEY_INTENT_CARRY_PUCK,
    HOCKEY_INTENT_CHANGE_PLAYER,
    HOCKEY_INTENT_NOOP,
    HOCKEY_INTENT_NORMAL_SHOOT,
    HOCKEY_INTENT_ONE_TIMER,
    HOCKEY_INTENT_PASS_START,
    HOCKEY_INTENT_POKE_CHECK,
    HOCKEY_INTENT_SLAPSHOT,
)


@dataclass(frozen=True)
class _Plan:
    name: str
    score: float
    target_x: float = 0.0
    target_y: float = 0.0
    intent: int = HOCKEY_INTENT_NOOP
    boost: bool = False
    pass_index: Optional[int] = None
    one_timer: bool = False
    shoot: bool = False
    slapshot: bool = False
    poke: bool = False
    change_player: bool = False


@dataclass(frozen=True)
class _ShotEval:
    score: float
    target_x: float
    target_y: float
    lane_clearance: float
    goalie_margin: float


class ClassicAIV2Model:
    """Option-scoring scripted controller for NHL94.

    The model keeps the same inference surface as Stable-Baselines models used
    in this project: ``predict`` for ordinary observations and
    ``predict_game_state`` for the NHL94 RAM-derived state object.
    """

    MIN_X = -112
    MAX_X = 112
    MIN_Y = -246
    MAX_Y = 246

    LOOKAHEAD_FRAMES = 42
    PLAYER_SPEED_ESTIMATE = 7.4
    PUCK_TIME_SCALE = 0.45
    PLAYER_TIME_SCALE = 0.20

    ATTACK_ZONE_Y = 88
    SLOT_MIN_X = -71
    SLOT_MAX_X = 71
    SLOT_ENTRY_Y = 122
    SLOT_SHOT_Y = 188
    NET_FRONT_Y = 214
    BEHIND_NET_Y = 236

    STICK_RADIUS = 14.0
    BODY_RADIUS = 8.0
    DEFAULT_GOALIE_RADIUS = 12.0
    PAD_STACK_RADIUS = 18.0

    PASS_COOLDOWN_FRAMES = 10
    ONE_TIMER_COOLDOWN_FRAMES = 18
    SWITCH_COOLDOWN_FRAMES = 12
    POKE_COOLDOWN_FRAMES = 8
    SHOT_COOLDOWN_FRAMES = 12
    SLAPSHOT_HOLD_FRAMES = 18

    TRACE_ENV_VAR = "CLASSIC_AI_V2_TRACE_PATH"

    def __init__(self, args=None, env=None):
        self.args = args
        self.env = env
        self._frame = 0
        self._last_action_preferences = np.zeros((1, self._action_size()), dtype=np.float32)
        self._last_plan = _Plan("init", 0.0)
        self._trace_path = os.environ.get(self.TRACE_ENV_VAR)

        self._pass_cooldown = 0
        self._one_timer_cooldown = 0
        self._switch_cooldown = 0
        self._poke_cooldown = 0
        self._shot_cooldown = 0
        self._slapshot_frames = 0

    def _uses_hockey_intents(self):
        return getattr(self.args, "action_type", "FILTERED").upper() == "HOCKEY_INTENT_DPAD"

    def _action_size(self):
        if self._uses_hockey_intents():
            return len(HOCKEY_INTENT_DPAD_ACTION_SPACE)
        return GameConsts.INPUT_MAX

    def predict(self, _state, deterministic=True):
        action = np.zeros((1, self._action_size()), dtype=np.int8)
        self._last_action_preferences = action.astype(np.float32)
        return action, None

    def predict_game_state(self, game_state, deterministic=True):
        self._frame += 1
        self._tick_cooldowns()

        plan = self._choose_plan(game_state)
        self._last_plan = plan
        if self._uses_hockey_intents():
            action = self._intent_action(game_state, plan)
        else:
            action = self._button_action(game_state, plan)

        return np.asarray([self._finalize(action)], dtype=np.int8)

    def get_action_preferences(self, _state=None):
        return self._last_action_preferences

    def learn(self, *args, **kwargs):
        raise NotImplementedError("ClassicAIV2 is inference-only. Use scripts/play.py to run it.")

    def save(self, *args, **kwargs):
        raise NotImplementedError("ClassicAIV2 does not produce a trainable checkpoint.")

    def _tick_cooldowns(self):
        self._pass_cooldown = max(0, self._pass_cooldown - 1)
        self._one_timer_cooldown = max(0, self._one_timer_cooldown - 1)
        self._switch_cooldown = max(0, self._switch_cooldown - 1)
        self._poke_cooldown = max(0, self._poke_cooldown - 1)
        self._shot_cooldown = max(0, self._shot_cooldown - 1)

    def _choose_plan(self, game_state):
        team = game_state.team1
        opponents = game_state.team2
        controlled = team.get_controlled_player()

        if team.player_haspuck:
            self._slapshot_frames = 0
            return self._plan_offense(game_state, controlled, team, opponents)

        if team.goalie_haspuck:
            self._slapshot_frames = 0
            return self._plan_goalie_puck(game_state, team, opponents)

        self._pass_cooldown = 0
        self._one_timer_cooldown = max(0, self._one_timer_cooldown - 1)
        self._slapshot_frames = 0
        return self._plan_no_puck(game_state, controlled, team, opponents)

    def _plan_offense(self, game_state, controlled, team, opponents):
        attack_sign = self._attack_sign(team, opponents)
        plans = []

        shot_eval = self._evaluate_shot(controlled, team, opponents, attack_sign, one_timer=False)
        if self._shot_cooldown <= 0:
            if shot_eval.score >= 118.0:
                plans.append(_Plan(
                    "open_net_shot",
                    shot_eval.score + 12.0,
                    shot_eval.target_x,
                    shot_eval.target_y,
                    intent=HOCKEY_INTENT_NORMAL_SHOOT,
                    shoot=True,
                ))
            elif shot_eval.score >= 102.0 and self._progress(self._y(controlled), attack_sign) >= self.SLOT_SHOT_Y:
                plans.append(_Plan(
                    "slot_shot",
                    shot_eval.score,
                    shot_eval.target_x,
                    shot_eval.target_y,
                    intent=HOCKEY_INTENT_NORMAL_SHOOT,
                    shoot=True,
                ))
            elif shot_eval.score >= 88.0 and self._can_hold_slapshot(controlled, opponents, attack_sign):
                plans.append(_Plan(
                    "space_slapshot",
                    shot_eval.score - 4.0,
                    shot_eval.target_x,
                    shot_eval.target_y,
                    intent=HOCKEY_INTENT_SLAPSHOT,
                    slapshot=True,
                ))

        one_timer_plan = self._one_timer_plan(controlled, team, opponents, attack_sign)
        if one_timer_plan is not None:
            plans.append(one_timer_plan)

        pass_plan = self._best_pass_plan(controlled, team, opponents, attack_sign)
        if pass_plan is not None:
            plans.append(pass_plan)

        carry_plan = self._best_carry_plan(controlled, team, opponents, attack_sign)
        plans.append(carry_plan)

        top_shelf_plan = self._top_shelf_plan(controlled, opponents, attack_sign)
        if top_shelf_plan is not None:
            plans.append(top_shelf_plan)

        return max(plans, key=lambda plan: plan.score)

    def _plan_goalie_puck(self, game_state, team, opponents):
        goalie = team.goalie
        attack_sign = self._attack_sign(team, opponents)
        pass_plan = self._best_pass_plan(goalie, team, opponents, attack_sign, goalie_pass=True)
        if pass_plan is not None and self._pass_cooldown <= 0:
            return pass_plan

        clear_y = self._signed_y(self.ATTACK_ZONE_Y + 16, attack_sign)
        return _Plan(
            "goalie_clear_lane",
            30.0,
            0.0,
            clear_y,
            intent=HOCKEY_INTENT_PASS_START,
            pass_index=0 if team.players else None,
        )

    def _plan_no_puck(self, game_state, controlled, team, opponents):
        puck = game_state.puck
        opponent_has_puck = bool(opponents.player_haspuck or opponents.goalie_haspuck)

        switch_plan = self._switch_plan(controlled, team, opponents, puck)
        if switch_plan is not None:
            return switch_plan

        if opponent_has_puck:
            return self._defensive_pressure_plan(controlled, team, opponents, puck)

        target_x, target_y, urgency = self._best_loose_puck_target(controlled, opponents, puck)
        boost = urgency > 28.0 and self._distance_xy(self._x(controlled), self._y(controlled), target_x, target_y) > 18.0
        return _Plan(
            "loose_puck_intercept",
            50.0 + urgency,
            target_x,
            target_y,
            intent=HOCKEY_INTENT_CARRY_PUCK,
            boost=boost,
        )

    def _switch_plan(self, controlled, team, opponents, puck):
        if self._switch_cooldown > 0 or len(team.players) <= 1:
            return None

        if self._distance_between(controlled, puck) <= 14.0:
            return None

        current_score = self._intercept_cost(controlled, opponents, puck)
        best_player = controlled
        best_score = current_score
        for player in team.players:
            score = self._intercept_cost(player, opponents, puck)
            if score < best_score:
                best_score = score
                best_player = player

        if best_player is controlled or best_score + 16.0 >= current_score:
            return None

        self._switch_cooldown = self.SWITCH_COOLDOWN_FRAMES
        return _Plan(
            "switch_to_best_interceptor",
            72.0 + current_score - best_score,
            self._x(best_player),
            self._y(best_player),
            intent=HOCKEY_INTENT_CHANGE_PLAYER,
            change_player=True,
        )

    def _defensive_pressure_plan(self, controlled, team, opponents, puck):
        carrier = opponents.get_controlled_player()
        attack_sign = self._attack_sign(opponents, team)
        carrier_progress = self._progress(self._y(carrier), attack_sign)
        own_goal_y = team.net.y

        target_x, target_y = self._carrier_stick_target(carrier, puck)
        slot_threat = carrier_progress >= self.ATTACK_ZONE_Y and self.SLOT_MIN_X <= self._x(carrier) <= self.SLOT_MAX_X
        if slot_threat:
            target_x = (self._x(carrier) * 0.65) + 0.0
            target_y = (self._y(carrier) * 0.65) + (own_goal_y * 0.35)

        distance_to_carrier = self._distance_between(controlled, carrier)
        poke = self._poke_cooldown <= 0 and distance_to_carrier <= 19.0 and self._is_facing_lane(controlled, target_x, target_y)
        if poke:
            self._poke_cooldown = self.POKE_COOLDOWN_FRAMES

        boost = distance_to_carrier > 24.0 or slot_threat
        return _Plan(
            "slot_pressure" if slot_threat else "carrier_pressure",
            72.0 if slot_threat else 58.0,
            target_x,
            target_y,
            intent=HOCKEY_INTENT_POKE_CHECK if poke else HOCKEY_INTENT_CARRY_PUCK,
            poke=poke,
            boost=boost,
        )

    def _best_loose_puck_target(self, controlled, opponents, puck):
        player_x = self._x(controlled)
        player_y = self._y(controlled)
        player_vx = self._vx(controlled)
        player_vy = self._vy(controlled)
        best_target = (self._x(puck), self._y(puck))
        best_error = float("inf")

        for frame_count in range(1, self.LOOKAHEAD_FRAMES + 1):
            puck_x, puck_y = self._predict_puck(frame_count, puck)
            projected_x = player_x + player_vx * min(frame_count, 8) * self.PLAYER_TIME_SCALE
            projected_y = player_y + player_vy * min(frame_count, 8) * self.PLAYER_TIME_SCALE
            distance = self._distance_xy(projected_x, projected_y, puck_x, puck_y)
            reachable = self.PLAYER_SPEED_ESTIMATE * frame_count
            error = abs(distance - reachable) + frame_count * 0.18
            if error < best_error:
                best_error = error
                best_target = (puck_x, puck_y)

        carrier = self._opponent_carrier(opponents)
        if carrier is not None:
            stick_x, stick_y = self._carrier_stick_target(carrier, puck)
            best_target = (best_target[0] * 0.45 + stick_x * 0.55, best_target[1] * 0.45 + stick_y * 0.55)

        urgency = max(0.0, 42.0 - best_error)
        return self._clip_x(best_target[0]), self._clip_y(best_target[1]), urgency

    def _intercept_cost(self, player, opponents, puck):
        player_x = self._x(player)
        player_y = self._y(player)
        player_vx = self._vx(player)
        player_vy = self._vy(player)
        best_cost = float("inf")

        for frame_count in range(1, self.LOOKAHEAD_FRAMES + 1):
            puck_x, puck_y = self._predict_puck(frame_count, puck)
            projected_x = player_x + player_vx * min(frame_count, 8) * self.PLAYER_TIME_SCALE
            projected_y = player_y + player_vy * min(frame_count, 8) * self.PLAYER_TIME_SCALE
            distance = self._distance_xy(projected_x, projected_y, puck_x, puck_y)
            cost = max(0.0, distance - self.PLAYER_SPEED_ESTIMATE * frame_count) + frame_count * 0.35
            best_cost = min(best_cost, cost)

        carrier = self._opponent_carrier(opponents)
        if carrier is not None:
            best_cost += self._distance_between(player, carrier) * 0.18
        return best_cost

    def _best_carry_plan(self, controlled, team, opponents, attack_sign):
        candidates = self._carry_candidates(controlled, opponents, attack_sign)
        best_plan = _Plan("carry", -float("inf"), self._x(controlled), self._y(controlled), intent=HOCKEY_INTENT_CARRY_PUCK)
        for target_x, target_y, name in candidates:
            score = self._carry_target_score(controlled, team, opponents, attack_sign, target_x, target_y)
            if score > best_plan.score:
                best_plan = _Plan(name, score, target_x, target_y, intent=HOCKEY_INTENT_CARRY_PUCK, boost=score < 58.0)
        return best_plan

    def _carry_candidates(self, controlled, opponents, attack_sign):
        current_progress = self._progress(self._y(controlled), attack_sign)
        current_x = self._x(controlled)
        candidates = []

        if current_progress < self.ATTACK_ZONE_Y:
            for lane_x in (-74, -42, -16, 16, 42, 74):
                candidates.append((lane_x, self._signed_y(max(self.ATTACK_ZONE_Y + 18, current_progress + 54), attack_sign), "carry_entry"))
            return candidates

        side = 1.0 if current_x >= 0 else -1.0
        candidates.extend([
            (0, self._signed_y(self.SLOT_SHOT_Y, attack_sign), "carry_slot"),
            (side * 68, self._signed_y(self.SLOT_SHOT_Y + 18, attack_sign), "lateral_goalie_pull"),
            (-side * 68, self._signed_y(self.SLOT_SHOT_Y + 10, attack_sign), "cross_slot_curl"),
            (side * 86, self._signed_y(self.ATTACK_ZONE_Y + 54, attack_sign), "wide_reset"),
            (-side * 48, self._signed_y(self.ATTACK_ZONE_Y + 68, attack_sign), "slot_bait_escape"),
        ])

        if self._nearest_point_distance(current_x, self._y(controlled), opponents.players) < 30.0:
            candidates.extend([
                (side * 82, self._signed_y(self.BEHIND_NET_Y, attack_sign), "behind_net_escape"),
                (-side * 82, self._signed_y(self.BEHIND_NET_Y - 10, attack_sign), "behind_net_reverse"),
            ])

        return candidates

    def _carry_target_score(self, controlled, team, opponents, attack_sign, target_x, target_y):
        progress = self._progress(target_y, attack_sign)
        current_progress = self._progress(self._y(controlled), attack_sign)
        opponent_group = self._players_and_goalie(opponents)
        space = self._nearest_point_distance(target_x, target_y, opponent_group)
        lane_space = self._lane_clearance((self._x(controlled), self._y(controlled)), (target_x, target_y), opponent_group)
        shot_eval = self._evaluate_virtual_shot(target_x, target_y, team, opponents, attack_sign, one_timer=False)
        travel = self._distance_xy(self._x(controlled), self._y(controlled), target_x, target_y)
        goalie = opponents.goalie
        goalie_distance = self._distance_xy(target_x, target_y, self._x(goalie), self._y(goalie))

        score = 22.0
        score += min(space, 58.0) * 0.80
        score += min(lane_space, 46.0) * 0.45
        score += max(0.0, progress - current_progress) * 0.20
        score += shot_eval.score * 0.38
        score -= travel * 0.11

        if progress >= self.SLOT_ENTRY_Y and self.SLOT_MIN_X <= target_x <= self.SLOT_MAX_X:
            score += 16.0
        if progress > self.NET_FRONT_Y and abs(target_x) < 42:
            score -= 28.0
        if progress >= self.BEHIND_NET_Y and abs(target_x) > 58:
            score += 10.0 if space < 30.0 else -8.0
        if goalie_distance < 28.0:
            score -= (28.0 - goalie_distance) * 2.5
        return score

    def _top_shelf_plan(self, controlled, opponents, attack_sign):
        progress = self._progress(self._y(controlled), attack_sign)
        if progress < 214 or abs(self._x(controlled)) > 38 or self._shot_cooldown > 0:
            return None
        if opponents.goalie.is_pad_stack:
            return None
        goalie_gap = abs(self._x(controlled) - self._x(opponents.goalie))
        if goalie_gap < 9.0 and not opponents.goalie.is_dive:
            return None
        return _Plan(
            "close_top_shelf",
            116.0 + goalie_gap,
            self._x(controlled),
            opponents.net.y,
            intent=HOCKEY_INTENT_NORMAL_SHOOT,
            shoot=True,
        )

    def _can_hold_slapshot(self, controlled, opponents, attack_sign):
        progress = self._progress(self._y(controlled), attack_sign)
        if progress < self.ATTACK_ZONE_Y + 8 or progress > self.NET_FRONT_Y:
            return False
        return self._nearest_point_distance(self._x(controlled), self._y(controlled), opponents.players) > 36.0

    def _best_pass_plan(self, passer, team, opponents, attack_sign, goalie_pass=False):
        if self._pass_cooldown > 0:
            return None

        controlled_index = int(getattr(team, "control", 0) or 0) - 1
        pass_options = []
        teammates = list(getattr(team, "players", []) or [])
        for player_index, teammate in enumerate(teammates):
            if not goalie_pass and player_index == controlled_index:
                continue
            pass_index = self._pass_index_for_player(team, teammate)
            if pass_index is None:
                continue

            lead_x, lead_y = self._lead_target(teammate, 8)
            lane_space = self._lane_clearance(
                (self._x(passer), self._y(passer)),
                (lead_x, lead_y),
                self._players_and_goalie(opponents),
            )
            if lane_space < 16.0:
                continue

            progress = self._progress(lead_y, attack_sign)
            if progress < -120.0 and not goalie_pass:
                continue

            receiver_space = self._nearest_point_distance(lead_x, lead_y, self._players_and_goalie(opponents))
            receiver_shot = self._evaluate_virtual_shot(lead_x, lead_y, team, opponents, attack_sign, one_timer=False)
            pass_distance = self._distance_xy(self._x(passer), self._y(passer), lead_x, lead_y)
            score = 18.0 + lane_space * 0.85 + receiver_space * 0.65 + receiver_shot.score * 0.58
            score += max(0.0, progress - self._progress(self._y(passer), attack_sign)) * 0.12
            score -= pass_distance * 0.12

            if goalie_pass:
                score += max(0.0, progress + 80.0) * 0.24
            if self._is_legal_one_timer_spot(lead_x, lead_y, attack_sign):
                score += 14.0
            if receiver_space < 20.0:
                score -= 28.0

            pass_options.append(_Plan(
                "goalie_outlet_pass" if goalie_pass else "value_pass",
                score,
                lead_x,
                lead_y,
                intent=HOCKEY_INTENT_PASS_START + pass_index,
                pass_index=pass_index,
            ))

        if not pass_options:
            return None
        best_plan = max(pass_options, key=lambda plan: plan.score)
        threshold = 44.0 if goalie_pass else 76.0
        return best_plan if best_plan.score >= threshold else None

    def _one_timer_plan(self, passer, team, opponents, attack_sign):
        if self._one_timer_cooldown > 0 or self._pass_cooldown > 0:
            return None
        if self._progress(self._y(passer), attack_sign) < self.ATTACK_ZONE_Y + 8:
            return None

        controlled_index = int(getattr(team, "control", 0) or 0) - 1
        best_plan = None
        for player_index, teammate in enumerate(team.players):
            if player_index == controlled_index:
                continue
            pass_index = self._pass_index_for_player(team, teammate)
            if pass_index is None:
                continue
            lead_x, lead_y = self._lead_target(teammate, 7)
            if not self._is_legal_one_timer_spot(lead_x, lead_y, attack_sign):
                continue

            lane_space = self._lane_clearance(
                (self._x(passer), self._y(passer)),
                (lead_x, lead_y),
                self._players_and_goalie(opponents),
            )
            if lane_space < 20.0:
                continue

            receiver_space = self._nearest_point_distance(lead_x, lead_y, self._players_and_goalie(opponents))
            shot_eval = self._evaluate_virtual_shot(lead_x, lead_y, team, opponents, attack_sign, one_timer=True)
            goalie_shift = abs(self._x(opponents.goalie) - lead_x) + abs(self._vx(opponents.goalie)) * 0.75
            lateral = abs(lead_x - self._x(passer))
            score = 42.0 + lane_space * 1.05 + receiver_space * 0.85 + shot_eval.score * 0.82
            score += lateral * 0.35 + goalie_shift * 0.22
            if score < 104.0:
                continue

            plan = _Plan(
                "one_timer_triangle",
                score,
                lead_x,
                lead_y,
                intent=HOCKEY_INTENT_ONE_TIMER,
                pass_index=pass_index,
                one_timer=True,
            )
            if best_plan is None or plan.score > best_plan.score:
                best_plan = plan

        return best_plan

    def _evaluate_shot(self, shooter, team, opponents, attack_sign, one_timer):
        return self._evaluate_virtual_shot(self._x(shooter), self._y(shooter), team, opponents, attack_sign, one_timer)

    def _evaluate_virtual_shot(self, shot_x, shot_y, team, opponents, attack_sign, one_timer):
        net_y = opponents.net.y
        target_points = [(0.0, net_y), (-13.0, net_y), (13.0, net_y)]
        best_eval = _ShotEval(-20.0, 0.0, net_y, 0.0, -99.0)
        shooter_progress = self._progress(shot_y, attack_sign)
        goalie = opponents.goalie
        goalie_radius = self._goalie_radius(goalie, one_timer)

        for target_x, target_y in target_points:
            lane_clearance = self._lane_clearance((shot_x, shot_y), (target_x, target_y), opponents.players)
            goalie_margin = self._line_clearance((shot_x, shot_y), (target_x, target_y), goalie) - goalie_radius
            distance_to_net = self._distance_xy(shot_x, shot_y, target_x, target_y)
            angle_width = 22.0 - abs(shot_x - target_x) * 0.12

            score = 18.0
            score += min(lane_clearance, 45.0) * 0.75
            score += goalie_margin * 2.2
            score += max(0.0, shooter_progress - self.ATTACK_ZONE_Y) * 0.28
            score += angle_width
            score -= max(0.0, distance_to_net - 190.0) * 0.18
            if self.SLOT_MIN_X <= shot_x <= self.SLOT_MAX_X and shooter_progress >= self.ATTACK_ZONE_Y:
                score += 18.0
            if shooter_progress >= 210 and abs(shot_x) < 42:
                score += 18.0
            if one_timer:
                score += 22.0
            if goalie.is_dive or goalie.is_pad_stack:
                score += 12.0
            if goalie_margin > 8.0:
                score += 18.0
            if lane_clearance < self.BODY_RADIUS + 4:
                score -= 35.0

            if score > best_eval.score:
                best_eval = _ShotEval(score, target_x, target_y, lane_clearance, goalie_margin)
        return best_eval

    def _goalie_radius(self, goalie, one_timer):
        if one_timer:
            return self.DEFAULT_GOALIE_RADIUS
        if getattr(goalie, "is_pad_stack", 0.0):
            return self.PAD_STACK_RADIUS
        if getattr(goalie, "is_dive", 0.0):
            return 14.0
        return max(self.DEFAULT_GOALIE_RADIUS, min(18.0, self.DEFAULT_GOALIE_RADIUS + abs(self._vx(goalie)) * 0.08))

    def _is_legal_one_timer_spot(self, x_pos, y_pos, attack_sign):
        progress = self._progress(y_pos, attack_sign)
        return self.ATTACK_ZONE_Y <= progress <= self.NET_FRONT_Y and self.SLOT_MIN_X <= x_pos <= self.SLOT_MAX_X

    def _pass_index_for_player(self, team, target_player):
        controlled_index = int(getattr(team, "control", 0) or 0) - 1
        teammates = [player for player_index, player in enumerate(team.players) if player_index != controlled_index]
        for teammate_index, teammate in enumerate(teammates[:4]):
            if teammate is target_player:
                return teammate_index
        return None

    def _lead_target(self, player, frame_count):
        lead_x = self._x(player) + self._vx(player) * frame_count * self.PLAYER_TIME_SCALE
        lead_y = self._y(player) + self._vy(player) * frame_count * self.PLAYER_TIME_SCALE
        return self._clip_x(lead_x), self._clip_y(lead_y)

    def _predict_puck(self, frame_count, puck):
        predicted_x = self._x(puck) + self._vx(puck) * frame_count * self.PUCK_TIME_SCALE
        predicted_y = self._y(puck) + self._vy(puck) * frame_count * self.PUCK_TIME_SCALE

        if predicted_x < self.MIN_X:
            predicted_x = self.MIN_X + (self.MIN_X - predicted_x) * 0.65
        elif predicted_x > self.MAX_X:
            predicted_x = self.MAX_X - (predicted_x - self.MAX_X) * 0.65

        if predicted_y < self.MIN_Y:
            predicted_y = self.MIN_Y + (self.MIN_Y - predicted_y) * 0.45
        elif predicted_y > self.MAX_Y:
            predicted_y = self.MAX_Y - (predicted_y - self.MAX_Y) * 0.45

        return predicted_x, predicted_y

    def _lane_clearance(self, start, end, players):
        if not players:
            return 999.0
        return min(self._line_clearance(start, end, player) for player in players)

    def _line_clearance(self, start, end, player):
        start_x, start_y = start
        end_x, end_y = end
        line_x = end_x - start_x
        line_y = end_y - start_y
        line_len_sq = max(1.0, line_x * line_x + line_y * line_y)
        player_x = self._x(player)
        player_y = self._y(player)
        projection = ((player_x - start_x) * line_x + (player_y - start_y) * line_y) / line_len_sq
        projection = max(0.0, min(1.0, projection))
        closest_x = start_x + line_x * projection
        closest_y = start_y + line_y * projection
        return self._distance_xy(player_x, player_y, closest_x, closest_y)

    def _carrier_stick_target(self, carrier, puck):
        direction_x = self._x(puck) - self._x(carrier)
        direction_y = self._y(puck) - self._y(carrier)
        direction_length = max(1.0, math.hypot(direction_x, direction_y))
        stick_x = self._x(carrier) + direction_x / direction_length * 10.0
        stick_y = self._y(carrier) + direction_y / direction_length * 10.0
        return stick_x + self._vx(carrier) * 0.65, stick_y + self._vy(carrier) * 0.65

    def _opponent_carrier(self, opponents):
        if opponents.player_haspuck or opponents.goalie_haspuck:
            return opponents.get_controlled_player()
        return None

    def _players_and_goalie(self, team):
        players = list(getattr(team, "players", []) or [])
        goalie = getattr(team, "goalie", None)
        if goalie is not None:
            players.append(goalie)
        return players

    def _is_facing_lane(self, player, target_x, target_y):
        orientation = int(getattr(player, "orientation", 0) or 0)
        angle = orientation * (2.0 * math.pi / 8.0)
        forward_x = math.cos(angle)
        forward_y = math.sin(angle)
        delta_x = target_x - self._x(player)
        delta_y = target_y - self._y(player)
        distance = max(1.0, math.hypot(delta_x, delta_y))
        return (delta_x / distance) * forward_x + (delta_y / distance) * forward_y > 0.35

    def _attack_sign(self, team, opponents):
        team_net_y = self._y(getattr(team, "net", None))
        opponent_net_y = self._y(getattr(opponents, "net", None))
        if opponent_net_y != team_net_y:
            return 1 if opponent_net_y > team_net_y else -1
        return 1

    def _progress(self, y_pos, attack_sign):
        return y_pos * attack_sign

    def _signed_y(self, progress, attack_sign):
        return self._clip_y(progress * attack_sign)

    def _x(self, obj):
        return float(getattr(obj, "x", 0) or 0)

    def _y(self, obj):
        return float(getattr(obj, "y", 0) or 0)

    def _vx(self, obj):
        return float(getattr(obj, "vx", 0) or 0)

    def _vy(self, obj):
        return float(getattr(obj, "vy", 0) or 0)

    def _clip_x(self, value):
        return float(np.clip(value, self.MIN_X, self.MAX_X))

    def _clip_y(self, value):
        return float(np.clip(value, self.MIN_Y, self.MAX_Y))

    def _distance_between(self, first, second):
        return self._distance_xy(self._x(first), self._y(first), self._x(second), self._y(second))

    def _distance_xy(self, first_x, first_y, second_x, second_y):
        return float(math.hypot(first_x - second_x, first_y - second_y))

    def _nearest_point_distance(self, x_pos, y_pos, players):
        player_list = list(players or [])
        if not player_list:
            return 999.0
        return min(self._distance_xy(x_pos, y_pos, self._x(player), self._y(player)) for player in player_list)

    def _intent_action(self, game_state, plan):
        action = np.zeros(len(HOCKEY_INTENT_DPAD_ACTION_SPACE), dtype=np.int8)
        action[0] = plan.intent
        controlled = game_state.team1.get_controlled_player()

        if plan.change_player:
            action[0] = HOCKEY_INTENT_CHANGE_PLAYER
        elif plan.poke:
            action[0] = HOCKEY_INTENT_POKE_CHECK
        elif plan.one_timer:
            action[0] = HOCKEY_INTENT_ONE_TIMER
            self._pass_cooldown = self.PASS_COOLDOWN_FRAMES
            self._one_timer_cooldown = self.ONE_TIMER_COOLDOWN_FRAMES
        elif plan.pass_index is not None:
            action[0] = HOCKEY_INTENT_PASS_START + plan.pass_index
            self._pass_cooldown = self.PASS_COOLDOWN_FRAMES
        elif plan.slapshot:
            action[0] = HOCKEY_INTENT_SLAPSHOT
            self._shot_cooldown = self.SHOT_COOLDOWN_FRAMES
        elif plan.shoot:
            action[0] = HOCKEY_INTENT_NORMAL_SHOOT
            self._shot_cooldown = self.SHOT_COOLDOWN_FRAMES

        if not plan.change_player:
            self._steer_intent(action, controlled, plan.target_x, plan.target_y)
        if plan.boost:
            action[5] = 1
        return action

    def _button_action(self, game_state, plan):
        action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        controlled = game_state.team1.get_controlled_player()

        if plan.change_player:
            action[GameConsts.INPUT_B] = 1
            return action

        self._steer_buttons(action, controlled, plan.target_x, plan.target_y)

        if plan.poke:
            action[GameConsts.INPUT_B] = 1
        elif plan.one_timer or plan.pass_index is not None:
            action[GameConsts.INPUT_B] = 1
            self._pass_cooldown = self.PASS_COOLDOWN_FRAMES
            if plan.one_timer:
                self._one_timer_cooldown = self.ONE_TIMER_COOLDOWN_FRAMES
        elif plan.slapshot:
            if self._slapshot_frames < self.SLAPSHOT_HOLD_FRAMES:
                action[GameConsts.INPUT_C] = 1
                self._slapshot_frames += 1
            else:
                self._slapshot_frames = 0
                self._shot_cooldown = self.SHOT_COOLDOWN_FRAMES
        elif plan.shoot:
            action[GameConsts.INPUT_C] = 1
            self._shot_cooldown = self.SHOT_COOLDOWN_FRAMES
        elif plan.boost:
            action[GameConsts.INPUT_C] = 1

        return action

    def _steer_buttons(self, action, player, target_x, target_y, deadzone=4.0):
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

    def _steer_intent(self, action, player, target_x, target_y, deadzone=4.0):
        delta_x = target_x - self._x(player)
        delta_y = target_y - self._y(player)
        if delta_y <= -deadzone:
            action[2] = 1
        elif delta_y >= deadzone:
            action[1] = 1
        if delta_x <= -deadzone:
            action[3] = 1
        elif delta_x >= deadzone:
            action[4] = 1

    def _trace_action(self, action):
        if not self._trace_path:
            return

        needs_header = not os.path.exists(self._trace_path)
        with open(self._trace_path, "a", encoding="utf-8") as trace_file:
            if needs_header:
                trace_file.write("frame,plan,score,target_x,target_y,action\n")
            trace_file.write(
                f"{self._frame},{self._last_plan.name},{self._last_plan.score:.3f},"
                f"{self._last_plan.target_x:.1f},{self._last_plan.target_y:.1f},"
                f"{'/'.join(str(int(value)) for value in action)}\n"
            )

    def _finalize(self, action):
        self._trace_action(action)
        preferences = action.astype(np.float32)
        self._last_action_preferences = preferences.reshape(1, -1)
        return action