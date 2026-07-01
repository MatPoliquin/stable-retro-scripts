"""ClassicAI v3: anti-chase, lane-first NHL94 scripted oracle."""

import os

from classic_ai_v2 import ClassicAIV2Model, _Plan
from game_wrappers.nhl94.nhl94_const import GameConsts
from game_wrappers.nhl94.nhl94_intents import (
    HOCKEY_INTENT_CARRY_PUCK,
    HOCKEY_INTENT_CHANGE_PLAYER,
    HOCKEY_INTENT_NOOP,
    HOCKEY_INTENT_NORMAL_SHOOT,
    HOCKEY_INTENT_ONE_TIMER,
    HOCKEY_INTENT_PASS_START,
    HOCKEY_INTENT_POKE_CHECK,
)


class ClassicAIV3Model(ClassicAIV2Model):
    """A stronger hand-built NHL94 controller than ClassicAIV2.

    V3 keeps the v2 mechanics engine, then adds deliberate anti-chase offense
    and lane-first defense. It should be treated as an inference oracle, not a
    trainable SB3 policy.
    """

    LOOKAHEAD_FRAMES = 54
    PLAYER_SPEED_ESTIMATE = 7.9

    PASS_COOLDOWN_FRAMES = 7
    ONE_TIMER_COOLDOWN_FRAMES = 14
    SWITCH_COOLDOWN_FRAMES = 9
    POKE_COOLDOWN_FRAMES = 7
    SHOT_COOLDOWN_FRAMES = 9

    TRACE_ENV_VAR = "CLASSIC_AI_V3_TRACE_PATH"

    MANUAL_ONE_TIMER_DELAY_FRAMES = 2
    MANUAL_ONE_TIMER_SHOT_FRAMES = 14
    PRESSURE_RELEASE_DISTANCE = 35.0
    DEFENSIVE_SLOT_Y = 184.0

    def __init__(self, args=None, env=None):
        super().__init__(args=args, env=env)
        self._trace_path = os.environ.get(self.TRACE_ENV_VAR)
        self._manual_one_timer_delay = 0
        self._manual_one_timer_frames = 0
        self._attack_memory = {
            "last_plan": "init",
            "pressure_side": 0.0,
            "cross_window": 0,
        }

    def learn(self, *args, **kwargs):
        raise NotImplementedError("ClassicAIV3 is inference-only. Use scripts/play.py to run it.")

    def save(self, *args, **kwargs):
        raise NotImplementedError("ClassicAIV3 does not produce a trainable checkpoint.")

    def _tick_cooldowns(self):
        super()._tick_cooldowns()
        self._attack_memory["cross_window"] = max(0, self._attack_memory["cross_window"] - 1)

    def _choose_plan(self, game_state):
        if not self._uses_hockey_intents() and self._manual_one_timer_frames > 0:
            if game_state.team2.player_haspuck or game_state.team2.goalie_haspuck:
                self._manual_one_timer_delay = 0
                self._manual_one_timer_frames = 0
            elif self._manual_one_timer_delay > 0:
                self._manual_one_timer_delay -= 1
                controlled = game_state.team1.get_controlled_player()
                return _Plan("manual_one_timer_wait", 180.0, self._x(controlled), self._y(controlled), intent=HOCKEY_INTENT_NOOP)
            else:
                self._manual_one_timer_frames -= 1
                controlled = game_state.team1.get_controlled_player()
                return _Plan("manual_one_timer_release", 240.0, self._x(controlled), self._y(controlled), intent=HOCKEY_INTENT_NORMAL_SHOOT, shoot=True)

        plan = super()._choose_plan(game_state)
        self._attack_memory["last_plan"] = plan.name
        return plan

    def _plan_offense(self, game_state, controlled, team, opponents):
        attack_sign = self._attack_sign(team, opponents)
        base_plan = super()._plan_offense(game_state, controlled, team, opponents)
        plans = [base_plan]

        pressure_plan = self._anti_chase_carry_plan(controlled, team, opponents, attack_sign)
        if pressure_plan is not None:
            plans.append(pressure_plan)

        release_pass = self._pressure_release_pass(controlled, team, opponents, attack_sign)
        if release_pass is not None:
            plans.append(release_pass)

        cross_one_timer = self._v3_one_timer_plan(controlled, team, opponents, attack_sign)
        if cross_one_timer is not None:
            plans.append(cross_one_timer)

        rebound_or_far_side = self._far_side_finish_plan(controlled, team, opponents, attack_sign)
        if rebound_or_far_side is not None:
            plans.append(rebound_or_far_side)

        return max(plans, key=lambda plan: plan.score)

    def _anti_chase_carry_plan(self, controlled, team, opponents, attack_sign):
        opponent_group = self._players_and_goalie(opponents)
        nearest_distance = self._nearest_point_distance(self._x(controlled), self._y(controlled), opponents.players)
        progress = self._progress(self._y(controlled), attack_sign)
        if nearest_distance > self.PRESSURE_RELEASE_DISTANCE or progress < self.ATTACK_ZONE_Y:
            return None

        nearest = min(opponents.players, key=lambda player: self._distance_between(controlled, player))
        escape_side = self._x(controlled) - self._x(nearest)
        if abs(escape_side) < 4.0:
            escape_side = self._x(controlled) if abs(self._x(controlled)) > 4.0 else 1.0
        escape_side = 1.0 if escape_side >= 0.0 else -1.0
        self._attack_memory["pressure_side"] = escape_side

        candidate_progress = min(self.BEHIND_NET_Y, max(progress + 14.0, self.SLOT_SHOT_Y))
        raw_candidates = [
            (-escape_side * 78.0, self._signed_y(self.SLOT_SHOT_Y + 8, attack_sign), "anti_chase_cross_slot"),
            (escape_side * 86.0, self._signed_y(candidate_progress, attack_sign), "anti_chase_wall_escape"),
            (-escape_side * 86.0, self._signed_y(self.BEHIND_NET_Y - 4, attack_sign), "anti_chase_behind_reverse"),
            (0.0, self._signed_y(self.SLOT_ENTRY_Y + 18, attack_sign), "anti_chase_high_reset"),
        ]

        best_plan = None
        for target_x, target_y, name in raw_candidates:
            lane_space = self._lane_clearance((self._x(controlled), self._y(controlled)), (target_x, target_y), opponent_group)
            if lane_space < 12.0:
                continue
            score = self._carry_target_score(controlled, team, opponents, attack_sign, target_x, target_y)
            score += (self.PRESSURE_RELEASE_DISTANCE - nearest_distance) * 2.4
            score += min(34.0, lane_space) * 0.55
            if "cross_slot" in name:
                score += 14.0
            if "behind" in name and progress > self.NET_FRONT_Y - 12:
                score += 18.0
            plan = _Plan(name, score, target_x, target_y, intent=HOCKEY_INTENT_CARRY_PUCK, boost=True)
            if best_plan is None or plan.score > best_plan.score:
                best_plan = plan
        return best_plan

    def _pressure_release_pass(self, passer, team, opponents, attack_sign):
        if self._pass_cooldown > 0:
            return None
        pressure = self._nearest_point_distance(self._x(passer), self._y(passer), opponents.players)
        if pressure > self.PRESSURE_RELEASE_DISTANCE:
            return None

        controlled_index = int(getattr(team, "control", 0) or 0) - 1
        best_plan = None
        for player_index, teammate in enumerate(team.players):
            if player_index == controlled_index:
                continue
            pass_index = self._pass_index_for_player(team, teammate)
            if pass_index is None:
                continue
            lead_x, lead_y = self._lead_target(teammate, 9)
            receiver_progress = self._progress(lead_y, attack_sign)
            if receiver_progress < self.ATTACK_ZONE_Y - 18:
                continue
            lane_space = self._lane_clearance((self._x(passer), self._y(passer)), (lead_x, lead_y), self._players_and_goalie(opponents))
            receiver_space = self._nearest_point_distance(lead_x, lead_y, self._players_and_goalie(opponents))
            if lane_space < 14.0 or receiver_space < 15.0:
                continue
            lateral_flip = (lead_x * self._x(passer)) < 0.0
            shot_eval = self._evaluate_virtual_shot(lead_x, lead_y, team, opponents, attack_sign, one_timer=False)
            score = 54.0 + lane_space * 1.05 + receiver_space * 0.72 + shot_eval.score * 0.52
            score += (self.PRESSURE_RELEASE_DISTANCE - pressure) * 1.8
            if lateral_flip:
                score += 18.0
            if self._is_legal_one_timer_spot(lead_x, lead_y, attack_sign):
                score += 16.0
            plan = _Plan("pressure_release_pass", score, lead_x, lead_y, intent=HOCKEY_INTENT_PASS_START + pass_index, pass_index=pass_index)
            if best_plan is None or plan.score > best_plan.score:
                best_plan = plan
        return best_plan if best_plan is not None and best_plan.score >= 82.0 else None

    def _v3_one_timer_plan(self, passer, team, opponents, attack_sign):
        if self._one_timer_cooldown > 0 or self._pass_cooldown > 0:
            return None
        if self._progress(self._y(passer), attack_sign) < self.ATTACK_ZONE_Y:
            return None

        controlled_index = int(getattr(team, "control", 0) or 0) - 1
        best_plan = None
        passer_pressure = self._nearest_point_distance(self._x(passer), self._y(passer), opponents.players)
        for player_index, teammate in enumerate(team.players):
            if player_index == controlled_index:
                continue
            pass_index = self._pass_index_for_player(team, teammate)
            if pass_index is None:
                continue
            lead_x, lead_y = self._lead_target(teammate, 7)
            if not self._is_legal_one_timer_spot(lead_x, lead_y, attack_sign):
                continue

            lane_space = self._lane_clearance((self._x(passer), self._y(passer)), (lead_x, lead_y), self._players_and_goalie(opponents))
            receiver_space = self._nearest_point_distance(lead_x, lead_y, self._players_and_goalie(opponents))
            if lane_space < 17.0 or receiver_space < 18.0:
                continue

            lateral = abs(lead_x - self._x(passer))
            shot_eval = self._evaluate_virtual_shot(lead_x, lead_y, team, opponents, attack_sign, one_timer=True)
            goalie = opponents.goalie
            goalie_lag = abs(self._x(goalie) - lead_x) + abs(self._vx(goalie)) * 0.9
            score = 58.0 + shot_eval.score * 0.90 + lane_space * 1.18 + receiver_space * 0.80
            score += lateral * 0.42 + goalie_lag * 0.25
            if passer_pressure < self.PRESSURE_RELEASE_DISTANCE:
                score += (self.PRESSURE_RELEASE_DISTANCE - passer_pressure) * 1.4
            if self._attack_memory["cross_window"] > 0:
                score += 12.0
            if score < 92.0:
                continue

            plan = _Plan("v3_cross_slot_one_timer", score, lead_x, lead_y, intent=HOCKEY_INTENT_ONE_TIMER, pass_index=pass_index, one_timer=True)
            if best_plan is None or plan.score > best_plan.score:
                best_plan = plan

        if best_plan is not None:
            self._attack_memory["cross_window"] = 18
        return best_plan

    def _far_side_finish_plan(self, controlled, team, opponents, attack_sign):
        if self._shot_cooldown > 0:
            return None
        progress = self._progress(self._y(controlled), attack_sign)
        if progress < self.SLOT_SHOT_Y - 8:
            return None

        goalie = opponents.goalie
        goalie_side = 1.0 if self._x(goalie) >= 0.0 else -1.0
        target_x = -goalie_side * 13.0
        shot_eval = self._evaluate_virtual_shot(self._x(controlled), self._y(controlled), team, opponents, attack_sign, one_timer=False)
        goalie_gap = abs(self._x(goalie) - self._x(controlled))
        lane_space = self._lane_clearance((self._x(controlled), self._y(controlled)), (target_x, opponents.net.y), opponents.players)
        if lane_space < 9.0:
            return None
        score = shot_eval.score + goalie_gap * 0.65 + lane_space * 0.35
        if opponents.goalie.is_dive or opponents.goalie.is_pad_stack:
            score += 18.0
        if score < 106.0:
            return None
        return _Plan("far_side_finish", score, target_x, opponents.net.y, intent=HOCKEY_INTENT_NORMAL_SHOOT, shoot=True)

    def _defensive_pressure_plan(self, controlled, team, opponents, puck):
        carrier = opponents.get_controlled_player()
        attack_sign = self._attack_sign(opponents, team)
        carrier_progress = self._progress(self._y(carrier), attack_sign)

        receiver_plan = self._cover_one_timer_receiver(controlled, team, opponents, carrier, attack_sign)
        lane_plan = self._protect_slot_lane(controlled, team, opponents, carrier, attack_sign)
        pressure_plan = super()._defensive_pressure_plan(controlled, team, opponents, puck)

        plans = [pressure_plan, lane_plan]
        if receiver_plan is not None:
            plans.append(receiver_plan)

        if carrier_progress >= self.ATTACK_ZONE_Y and abs(self._x(carrier)) > 76.0:
            lane_plan = _Plan(
                "do_not_chase_corner_slot_lock",
                lane_plan.score + 22.0,
                lane_plan.target_x,
                lane_plan.target_y,
                intent=lane_plan.intent,
                boost=lane_plan.boost,
                poke=lane_plan.poke,
            )
            plans.append(lane_plan)

        return max(plans, key=lambda plan: plan.score)

    def _cover_one_timer_receiver(self, controlled, team, opponents, carrier, attack_sign):
        best_target = None
        best_score = -float("inf")
        carrier_point = (self._x(carrier), self._y(carrier))
        for receiver in opponents.players:
            if receiver is carrier:
                continue
            receiver_x, receiver_y = self._lead_target(receiver, 5)
            progress = self._progress(receiver_y, attack_sign)
            if progress < self.ATTACK_ZONE_Y or progress > self.NET_FRONT_Y + 14:
                continue
            if abs(receiver_x) > 82.0:
                continue
            lane_space = self._line_clearance(carrier_point, (receiver_x, receiver_y), controlled)
            shot_lane = self._line_clearance((receiver_x, receiver_y), (0.0, team.net.y), controlled)
            distance = self._distance_xy(self._x(controlled), self._y(controlled), receiver_x, receiver_y)
            score = 78.0 - distance * 0.35
            score += max(0.0, 22.0 - lane_space) * 1.8
            score += max(0.0, 24.0 - shot_lane) * 1.6
            score += max(0.0, 52.0 - abs(receiver_x)) * 0.20
            if score > best_score:
                best_score = score
                best_target = (receiver_x, receiver_y)

        if best_target is None or best_score < 64.0:
            return None
        poke = self._poke_cooldown <= 0 and self._distance_xy(self._x(controlled), self._y(controlled), best_target[0], best_target[1]) <= 18.0
        if poke:
            self._poke_cooldown = self.POKE_COOLDOWN_FRAMES
        return _Plan("cover_one_timer_receiver", best_score, best_target[0], best_target[1], intent=HOCKEY_INTENT_POKE_CHECK if poke else HOCKEY_INTENT_CARRY_PUCK, poke=poke, boost=True)

    def _protect_slot_lane(self, controlled, team, opponents, carrier, attack_sign):
        own_net_y = team.net.y
        carrier_x = self._x(carrier)
        carrier_y = self._y(carrier)
        slot_y = self._signed_y(self.DEFENSIVE_SLOT_Y, attack_sign)
        target_x = carrier_x * 0.42
        target_y = carrier_y * 0.42 + own_net_y * 0.34 + slot_y * 0.24

        carrier_distance = self._distance_between(controlled, carrier)
        lane_distance = self._line_clearance((carrier_x, carrier_y), (0.0, own_net_y), controlled)
        poke = self._poke_cooldown <= 0 and carrier_distance <= 18.0 and self._is_facing_lane(controlled, carrier_x, carrier_y)
        if poke:
            self._poke_cooldown = self.POKE_COOLDOWN_FRAMES
        score = 72.0 + max(0.0, 28.0 - lane_distance) * 1.8 - carrier_distance * 0.05
        return _Plan("protect_slot_lane", score, self._clip_x(target_x), self._clip_y(target_y), intent=HOCKEY_INTENT_POKE_CHECK if poke else HOCKEY_INTENT_CARRY_PUCK, poke=poke, boost=True)

    def _switch_plan(self, controlled, team, opponents, puck):
        if self._switch_cooldown > 0 or len(team.players) <= 1:
            return None

        if opponents.player_haspuck or opponents.goalie_haspuck:
            carrier = opponents.get_controlled_player()
            attack_sign = self._attack_sign(opponents, team)
            target_x = carrier.x * 0.42
            target_y = self._clip_y(carrier.y * 0.42 + team.net.y * 0.34 + self._signed_y(self.DEFENSIVE_SLOT_Y, attack_sign) * 0.24)
            current = self._distance_xy(self._x(controlled), self._y(controlled), target_x, target_y)
            best_player = controlled
            best_distance = current
            for player in team.players:
                distance = self._distance_xy(self._x(player), self._y(player), target_x, target_y)
                distance += self._distance_between(player, carrier) * 0.18
                if distance < best_distance:
                    best_distance = distance
                    best_player = player
            if best_player is not controlled and best_distance + 16.0 < current:
                self._switch_cooldown = self.SWITCH_COOLDOWN_FRAMES
                return _Plan("switch_to_lane_guard", 88.0 + current - best_distance, self._x(best_player), self._y(best_player), intent=HOCKEY_INTENT_CHANGE_PLAYER, change_player=True)

        return super()._switch_plan(controlled, team, opponents, puck)

    def _button_action(self, game_state, plan):
        action = super()._button_action(game_state, plan)
        if plan.one_timer and not self._uses_hockey_intents():
            self._manual_one_timer_delay = self.MANUAL_ONE_TIMER_DELAY_FRAMES
            self._manual_one_timer_frames = self.MANUAL_ONE_TIMER_SHOT_FRAMES
        return action

    def _intent_action(self, game_state, plan):
        action = super()._intent_action(game_state, plan)
        if plan.name == "pressure_release_pass" and plan.pass_index is not None:
            action[0] = HOCKEY_INTENT_PASS_START + plan.pass_index
        return action