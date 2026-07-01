import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from game_wrappers.nhl94.nhl94_gamestate import NHL94GameState
from game_wrappers.nhl94.nhl94_mi import init_model, set_model_input


def _goalie_stats(prefix):
    return {
        f"{prefix}goalie_glove_left": 4,
        f"{prefix}goalie_glove_right": 4,
        f"{prefix}goalie_stick_left": 4,
        f"{prefix}goalie_stick_right": 4,
        f"{prefix}goalie_puck_control": 4,
        f"{prefix}goalie_agility": 4,
        f"{prefix}goalie_speed": 4,
        f"{prefix}goalie_passing": 4,
        f"{prefix}goalie_endurance": 4,
        f"{prefix}goalie_weight": 4,
    }


def _base_info(*, team1_control_index=0, puck_owner=-1):
    player_positions = {
        "p1": [(-20, 40), (30, 70)],
        "p2": [(-25, -40), (35, -70)],
    }
    info = {
        "period": 1,
        "time": 300,
        "puck_x": 5,
        "puck_y": 55,
        "puck_vel_x": 2,
        "puck_vel_y": -1,
        "puck_owner": puck_owner,
        "shot_player": -1,
        "pass_dir": 0,
        "pass_speed": 0,
        "goalie_chk_body": 12,
        "sflags": 0,
        "sflags2": 0,
        "ba_ps_flags": 0,
        "word_ffc2f6": 0,
        "word_ffc2f8": 0,
        "word_ffc2fa": 0,
        "fullstar_x": 999,
        "fullstar_y": 999,
        "p2_fullstar_x": 999,
        "p2_fullstar_y": 999,
        "p1_score": 0,
        "p1_shots": 0,
        "p1_bodychecks": 0,
        "p1_attackzone": 0,
        "p1_faceoffwon": 0,
        "p1_passing": 0,
        "p1_onetimer": 0,
        "p2_score": 0,
        "p2_shots": 0,
        "p2_bodychecks": 0,
        "p2_attackzone": 0,
        "p2_faceoffwon": 0,
        "p2_passing": 0,
        "p2_onetimer": 0,
        "g1_x": 0,
        "g1_y": 120,
        "g1_vel_x": 0,
        "g1_vel_y": 0,
        "g1_anim": 0,
        "g1_anim_frame": 0,
        "g1_state_flags": 0,
        "g2_x": 0,
        "g2_y": -120,
        "g2_vel_x": 0,
        "g2_vel_y": 0,
        "g2_anim": 0,
        "g2_anim_frame": 0,
        "g2_state_flags": 0,
        **_goalie_stats("g1_"),
        **_goalie_stats("g2_"),
    }

    for prefix, positions in player_positions.items():
        empty_x, empty_y = positions[team1_control_index] if prefix == "p1" else positions[0]
        info[f"{prefix}_emptystar_x"] = empty_x
        info[f"{prefix}_emptystar_y"] = empty_y

        for index, (x_pos, y_pos) in enumerate(positions):
            player_prefix = prefix if index == 0 else f"{prefix}_{index + 1}"
            info[f"{player_prefix}_x"] = x_pos
            info[f"{player_prefix}_y"] = y_pos
            info[f"{player_prefix}_vel_x"] = index + 1
            info[f"{player_prefix}_vel_y"] = -(index + 1)
            info[f"{player_prefix}_ori"] = index
            info[f"{player_prefix}_anim"] = 0
            info[f"{player_prefix}_anim_frame"] = 0
            info[f"{player_prefix}_state_flags"] = 0

    return info


def _state(*, team1_control_index=0, puck_owner=-1):
    game_state = NHL94GameState(2)
    game_state.BeginFrame(
        _base_info(team1_control_index=team1_control_index, puck_owner=puck_owner),
        [0, 0, 0, 0, 0, 0],
    )
    return game_state


class NHL94ModelInputTests(unittest.TestCase):
    def test_fixed_slots_do_not_reorder_when_control_changes(self):
        config = {
            "schema_version": 2,
            "strict": True,
            "groups": {
                "friendly_player": ["x", "y", "rel_puck_x", "rel_puck_y", "is_controlled"],
                "friendly_goalie": [],
                "opponent_player": [],
                "opponent_goalie": [],
                "puck": [],
                "goalie_stats": [],
                "possession": [],
                "net": [],
                "buttons": [],
                "hidden_state": [],
            },
        }

        player0_controlled = set_model_input(_state(team1_control_index=0), config)
        player1_controlled = set_model_input(_state(team1_control_index=1), config)

        self.assertEqual(len(player0_controlled), init_model(2, config))
        self.assertEqual(player0_controlled[0:4], player1_controlled[0:4])
        self.assertEqual(player0_controlled[5:9], player1_controlled[5:9])
        self.assertEqual(player0_controlled[4], 1.0)
        self.assertEqual(player0_controlled[9], 0.0)
        self.assertEqual(player1_controlled[4], 0.0)
        self.assertEqual(player1_controlled[9], 1.0)

    def test_possession_flags_are_entity_local(self):
        config = {
            "schema_version": 2,
            "strict": True,
            "groups": {
                "friendly_player": ["has_puck"],
                "friendly_goalie": ["has_puck"],
                "opponent_player": ["has_puck"],
                "opponent_goalie": ["has_puck"],
                "puck": [],
                "goalie_stats": [],
                "possession": ["team1_has_puck", "team2_has_puck", "puck_loose"],
                "net": [],
                "buttons": [],
                "hidden_state": [],
            },
        }

        team1_second_skater_has_puck = set_model_input(_state(puck_owner=1), config)
        self.assertEqual(team1_second_skater_has_puck, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

        team2_goalie_has_puck = set_model_input(_state(puck_owner=11), config)
        self.assertEqual(team2_goalie_has_puck, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0))

        loose_puck = set_model_input(_state(puck_owner=-1), config)
        self.assertEqual(loose_puck, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()