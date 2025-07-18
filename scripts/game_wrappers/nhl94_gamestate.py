"""
NHL94 Game State
"""

import math
from game_wrappers.nhl94_const import GameConsts
from typing import Dict, Any
from copy import deepcopy
from dataclasses import dataclass
import time

@dataclass
class Player:
    x: int = 0
    y: int = 0
    vx: int = 0
    vy: int = 0
    orientation: int = 0  # New field for orientation (0-7)
    ori_x: float = 0.0    # Normalized x component of orientation
    ori_y: float = 0.0    # Normalized y component of orientation
    rel_puck_x: float = 0.0  # Relative puck x position
    rel_puck_y: float = 0.0  # Relative puck y position
    rel_puck_vx: float = 0.0 # Relative puck x velocity
    rel_puck_vy: float = 0.0 # Relative puck y velocity
    rel_controlled_x: float = 0.0  # New: Relative to controlled player x position
    rel_controlled_y: float = 0.0  # New: Relative to controlled player y position
    rel_controlled_vx: float = 0.0 # New: Relative to controlled player x velocity
    rel_controlled_vy: float = 0.0 # New: Relative to controlled player y velocity
    dist_to_controlled: float = 0.0  # New: Distance to controlled player
    dist_to_controlled_opp: float = 0.0  # New: Distance to controlled opponent player
    dist_to_puck: float = 0.0  # New: Distance to puck
    passing_lane_clear: bool = False

    def debug_print(self, prefix="Player"):
        print(f"{prefix} - x: {self.x}, y: {self.y}, vx: {self.vx}, vy: {self.vy}, "
            f"orientation: {self.orientation}, ori_vec: ({self.ori_x:.2f}, {self.ori_y:.2f})\n"
            f"  rel_puck: ({self.rel_puck_x:.1f}, {self.rel_puck_y:.1f}) "
            f"rel_puck_vel: ({self.rel_puck_vx:.1f}, {self.rel_puck_vy:.1f})\n"
            f"  rel_controlled: ({self.rel_controlled_x:.1f}, {self.rel_controlled_y:.1f}) "
            f"rel_controlled_vel: ({self.rel_controlled_vx:.1f}, {self.rel_controlled_vy:.1f})\n"
            f"  dist_to_controlled: {self.dist_to_controlled:.1f}, "
            f"dist_to_controlled_opp: {self.dist_to_controlled_opp:.1f}, "
            f"dist_to_puck: {self.dist_to_puck:.1f}\n"
            f"  passing_lane_clear: {self.passing_lane_clear}")

@dataclass
class Stats:
    score: int = 0
    shots: int = 0
    bodychecks: int = 0
    attackzone: int = 0
    faceoffwon: int = 0
    passing: int = 0
    onetimer: int = 0
    fullstar_x: int = 0
    fullstar_y: int = 0
    emptystar_x: int = 0
    emptystar_y: int = 0

    def debug_print(self, prefix="Stats"):
        print(f"{prefix} - score: {self.score}, shots: {self.shots}, bodychecks: {self.bodychecks}, "
              f"attackzone: {self.attackzone}, faceoffwon: {self.faceoffwon}, passing: {self.passing}, onetimer: {self.onetimer},"
              f"fullstar: ({self.fullstar_x},{self.fullstar_y}), emptystar: ({self.emptystar_x},{self.emptystar_y})")

class Team():
    HAS_PUCK_TRESHOLD = 3

    def __init__(self, controller: int, num_players: int):
        self.controller = controller
        self.stats = Stats()
        self.last_stats = Stats()
        self.num_players = num_players

        self.players = [Player() for _ in range(num_players)]
        self.goalie = Player()

        self.ram_var_prefix = 'p1_' if controller == 1 else 'p2_'
        self.ram_var_goalie_prefix = 'g1_' if controller == 1 else 'g2_'
        self.control = 0
        self.player_haspuck = False
        self.goalie_haspuck = False

        # for model input
        self.nz_players = [Player() for _ in range(num_players)]
        self.nz_goalie = Player()
        self.nz_player_haspuck = 0.0
        self.nz_goalie_haspuck = 0.0

    def has_puck(self, pos_x, pos_y):
        return (abs(pos_x - self.stats.fullstar_x) < self.HAS_PUCK_TRESHOLD and abs(pos_y - self.stats.fullstar_y) < self.HAS_PUCK_TRESHOLD)

    def has_control(self, pos_x, pos_y):
        return (abs(pos_x - self.stats.emptystar_x) < self.HAS_PUCK_TRESHOLD and abs(pos_y - self.stats.emptystar_y) < self.HAS_PUCK_TRESHOLD)

    def begin_frame(self, info: Dict[str, Any], puck_x: int, puck_y: int, puck_vx: int, puck_vy: int) -> None:
        # General state
        self.stats.score = info.get(f"{self.ram_var_prefix}score")
        self.stats.shots = info.get(f"{self.ram_var_prefix}shots")
        self.stats.bodychecks = info.get(f"{self.ram_var_prefix}bodychecks")
        self.stats.attackzone = info.get(f"{self.ram_var_prefix}attackzone")
        self.stats.faceoffwon = info.get(f"{self.ram_var_prefix}faceoffwon")
        self.stats.passing = info.get(f"{self.ram_var_prefix}passing")
        self.stats.onetimer = info.get(f"{self.ram_var_prefix}onetimer")

        # special case for team 1 as the stable-retro ram var name don't have the prefix
        if self.controller == 1:
            self.stats.fullstar_x = info.get(f"fullstar_x")
            self.stats.fullstar_y = info.get(f"fullstar_y")
        else:
            self.stats.fullstar_x = info.get(f"{self.ram_var_prefix}fullstar_x")
            self.stats.fullstar_y = info.get(f"{self.ram_var_prefix}fullstar_y")

        self.stats.emptystar_x = info.get(f"{self.ram_var_prefix}emptystar_x", 0)
        self.stats.emptystar_y = info.get(f"{self.ram_var_prefix}emptystar_y", 0)

        # Goalie
        self.goalie.x = info.get(f"{self.ram_var_goalie_prefix}x")
        self.goalie.y = info.get(f"{self.ram_var_goalie_prefix}y")
        self.goalie.vx = info.get(f"{self.ram_var_goalie_prefix}vel_x", 0)
        self.goalie.vy = info.get(f"{self.ram_var_goalie_prefix}vel_y", 0)

        # Players
        for p in range(0, self.num_players):
            if p == 0:
                self.players[p].x = info.get(f"{self.ram_var_prefix}x")
                self.players[p].y = info.get(f"{self.ram_var_prefix}y")
                self.players[p].vx = info.get(f"{self.ram_var_prefix}vel_x")
                self.players[p].vy = info.get(f"{self.ram_var_prefix}vel_y")
                self.players[p].orientation = info.get(f"{self.ram_var_prefix}ori", 0)
            else:
                pi = p + 1
                self.players[p].x = info.get(f"{self.ram_var_prefix}{pi}_x")
                self.players[p].y = info.get(f"{self.ram_var_prefix}{pi}_y")
                self.players[p].vx = info.get(f"{self.ram_var_prefix}{pi}_vel_x")
                self.players[p].vy = info.get(f"{self.ram_var_prefix}{pi}_vel_y")
                self.players[p].orientation = info.get(f"{self.ram_var_prefix}{pi}_ori", 0)

            # Convert orientation to vector
            angle = self.players[p].orientation * (2 * math.pi / 8)
            self.players[p].ori_x = math.cos(angle)
            self.players[p].ori_y = math.sin(angle)

            # Calculate relative puck position and velocity
            self.players[p].rel_puck_x = puck_x - self.players[p].x
            self.players[p].rel_puck_y = puck_y - self.players[p].y
            self.players[p].rel_puck_vx = puck_vx - self.players[p].vx
            self.players[p].rel_puck_vy = puck_vy - self.players[p].vy

        # Calculate relative puck position for goalie
        self.goalie.rel_puck_x = puck_x - self.goalie.x
        self.goalie.rel_puck_y = puck_y - self.goalie.y
        self.goalie.rel_puck_vx = puck_vx - self.goalie.vx
        self.goalie.rel_puck_vy = puck_vy - self.goalie.vy

        # Knowing if the player has the puck is tricky since the fullstar in the game is not aligned with the player every frame
        # There is an offset of up to 2 sometimes
        self.player_haspuck = False
        for p in range(0, self.num_players):
            if self.has_puck(self.players[p].x, self.players[p].y):
                self.player_haspuck = True

        self.goalie_haspuck = self.has_puck(self.goalie.x, self.goalie.y)

        # Check which player is being controlled
        if self.goalie_haspuck:
            self.control = 0
        elif self.player_haspuck:
            for p in range(0, self.num_players):
                if self.has_puck(self.players[p].x, self.players[p].y):
                    self.control = p + 1
        else:
            for p in range(0, self.num_players):
                if self.has_control(self.players[p].x, self.players[p].y):
                    self.control = p + 1

        # Calculate relative positions to controlled player
        controlled_x = self.goalie.x if self.control == 0 else self.players[self.control-1].x
        controlled_y = self.goalie.y if self.control == 0 else self.players[self.control-1].y
        controlled_vx = self.goalie.vx if self.control == 0 else self.players[self.control-1].vx
        controlled_vy = self.goalie.vy if self.control == 0 else self.players[self.control-1].vy

        for p in range(0, self.num_players):
            self.players[p].rel_controlled_x = self.players[p].x - controlled_x
            self.players[p].rel_controlled_y = self.players[p].y - controlled_y
            self.players[p].rel_controlled_vx = self.players[p].vx - controlled_vx
            self.players[p].rel_controlled_vy = self.players[p].vy - controlled_vy

        # For goalie
        self.goalie.rel_controlled_x = self.goalie.x - controlled_x
        self.goalie.rel_controlled_y = self.goalie.y - controlled_y
        self.goalie.rel_controlled_vx = self.goalie.vx - controlled_vx
        self.goalie.rel_controlled_vy = self.goalie.vy - controlled_vy

        # Calculate distances to controlled player and puck
        for p in range(0, self.num_players):
            self.players[p].dist_to_controlled = GameConsts.Distance(
                (self.players[p].x, self.players[p].y),
                (controlled_x, controlled_y)
            )
            self.players[p].dist_to_puck = GameConsts.Distance(
                (self.players[p].x, self.players[p].y),
                (puck_x, puck_y)
            )

        # For goalie
        self.goalie.dist_to_controlled = GameConsts.Distance(
            (self.goalie.x, self.goalie.y),
            (controlled_x, controlled_y)
        )
        self.goalie.dist_to_puck = GameConsts.Distance(
            (self.goalie.x, self.goalie.y),
            (puck_x, puck_y)
        )

        # Normalize for model input
        for p in range(0, self.num_players):
            self.nz_players[p].x = self.players[p].x / GameConsts.MAX_PLAYER_X
            self.nz_players[p].y = self.players[p].y / GameConsts.MAX_PLAYER_Y
            self.nz_players[p].vx = self.players[p].vx / GameConsts.MAX_VEL_XY
            self.nz_players[p].vy = self.players[p].vy / GameConsts.MAX_VEL_XY
            # Orientation vector is already normalized (-1 to 1)
            self.nz_players[p].ori_x = self.players[p].ori_x
            self.nz_players[p].ori_y = self.players[p].ori_y
            # Normalize relative puck position and velocity
            self.nz_players[p].rel_puck_x = self.players[p].rel_puck_x / GameConsts.MAX_PUCK_X
            self.nz_players[p].rel_puck_y = self.players[p].rel_puck_y / GameConsts.MAX_PUCK_Y
            self.nz_players[p].rel_puck_vx = self.players[p].rel_puck_vx / GameConsts.MAX_VEL_XY
            self.nz_players[p].rel_puck_vy = self.players[p].rel_puck_vy / GameConsts.MAX_VEL_XY
            # Normalize relative controlled player position and velocity
            self.nz_players[p].rel_controlled_x = self.players[p].rel_controlled_x / GameConsts.MAX_PLAYER_X
            self.nz_players[p].rel_controlled_y = self.players[p].rel_controlled_y / GameConsts.MAX_PLAYER_Y
            self.nz_players[p].rel_controlled_vx = self.players[p].rel_controlled_vx / GameConsts.MAX_VEL_XY
            self.nz_players[p].rel_controlled_vy = self.players[p].rel_controlled_vy / GameConsts.MAX_VEL_XY
            # Normalize distances
            self.nz_players[p].dist_to_controlled = self.players[p].dist_to_controlled / GameConsts.MAX_PLAYER_X
            self.nz_players[p].dist_to_puck = self.players[p].dist_to_puck / GameConsts.MAX_PUCK_X

        self.nz_goalie.x = self.goalie.x / GameConsts.MAX_PLAYER_X
        self.nz_goalie.y = self.goalie.y / GameConsts.MAX_PLAYER_Y
        self.nz_goalie.vx = self.goalie.vx / GameConsts.MAX_VEL_XY
        self.nz_goalie.vy = self.goalie.vy / GameConsts.MAX_VEL_XY
        # Normalize relative puck position and velocity for goalie
        self.nz_goalie.rel_puck_x = self.goalie.rel_puck_x / GameConsts.MAX_PUCK_X
        self.nz_goalie.rel_puck_y = self.goalie.rel_puck_y / GameConsts.MAX_PUCK_Y
        self.nz_goalie.rel_puck_vx = self.goalie.rel_puck_vx / GameConsts.MAX_VEL_XY
        self.nz_goalie.rel_puck_vy = self.goalie.rel_puck_vy / GameConsts.MAX_VEL_XY
        # Normalize relative controlled player position and velocity for goalie
        self.nz_goalie.rel_controlled_x = self.goalie.rel_controlled_x / GameConsts.MAX_PLAYER_X
        self.nz_goalie.rel_controlled_y = self.goalie.rel_controlled_y / GameConsts.MAX_PLAYER_Y
        self.nz_goalie.rel_controlled_vx = self.goalie.rel_controlled_vx / GameConsts.MAX_VEL_XY
        self.nz_goalie.rel_controlled_vy = self.goalie.rel_controlled_vy / GameConsts.MAX_VEL_XY
        # Normalize distances for goalie
        self.nz_goalie.dist_to_controlled = self.goalie.dist_to_controlled / GameConsts.MAX_PLAYER_X
        self.nz_goalie.dist_to_puck = self.goalie.dist_to_puck / GameConsts.MAX_PUCK_X

        # 0.0 and 1.0 switched around due to current models trained that way
        self.nz_player_haspuck = 1.0 if self.player_haspuck else 0.0
        self.nz_goalie_haspuck = 1.0 if self.goalie_haspuck else 0.0

    def end_frame(self) -> None:
        self.last_stats = deepcopy(self.stats)

    def debug_print(self):
        print(f"\nTeam controller: {self.controller}")
        print(f"Team player prefix: {self.ram_var_prefix}")
        self.stats.debug_print("Stats")
        self.last_stats.debug_print("Last Stats")
        print(f"Number of players: {self.num_players}")
        print(f"Control: {self.control}")
        print(f"Player has puck: {self.player_haspuck}")
        print(f"Goalie has puck: {self.goalie_haspuck}")

        # Print all players with full details
        for idx, player in enumerate(self.players):
            player.debug_print(f"Player {idx}")

        # Print goalie
        self.goalie.debug_print("Goalie")

        # Print normalized values
        print("\nNormalized values:")
        print(f"NZ Player has puck: {self.nz_player_haspuck}")
        print(f"NZ Goalie has puck: {self.nz_goalie_haspuck}")
        for idx, player in enumerate(self.nz_players):
            print(f"NZ Player {idx}: "
                f"pos=({player.x:.2f},{player.y:.2f}) "
                f"vel=({player.vx:.2f},{player.vy:.2f}) "
                f"ori=({player.ori_x:.2f},{player.ori_y:.2f})")
        self.nz_goalie.debug_print("NZ Goalie")


class NHL94GameState():
    def __init__(self, numPlayers):
        self.team1 = Team(1, numPlayers)
        self.team2 = Team(2, numPlayers)
        self.puck = Player()
        self.time = 0
        self.last_time = 0
        self.numPlayers = numPlayers

        self.action = [0] * 6 # Up, Down, Left, Right, B, C

        # Slapshot tracking
        self.slapshot_frames_held = 0
        self.SLAPSHOT_HOLD_FRAMES = 60  # Max frames to hold for slapshot

        #For model input
        self.nz_puck = Player()

    # Flip the variables
    def Flip(self):
        self.team1, self.team2 = self.team2, self.team1
        return

    def _is_passing_lane_clear(self, start_pos, end_pos, opponents):
        """Check if a straight-line path between two points is obstructed."""
        for opponent in opponents:
            if self._line_intersects_circle(start_pos, end_pos, (opponent.x, opponent.y), radius=10):
                return False
        return True

    def _line_intersects_circle(self, start, end, circle_center, radius):
        """Check if a line segment from 'start' to 'end' intersects a circle.

        Args:
            start: Tuple (x, y) of line segment start point
            end: Tuple (x, y) of line segment end point
            circle_center: Tuple (x, y) of circle center
            radius: Radius of the circle

        Returns:
            bool: True if the line segment intersects the circle
        """
        # Vector from start to end
        line_vec = (end[0] - start[0], end[1] - start[1])
        # Vector from start to circle center
        circle_vec = (circle_center[0] - start[0], circle_center[1] - start[1])

        # Length of line segment squared
        line_len_sq = line_vec[0]**2 + line_vec[1]**2

        # Projection of circle_vec onto line_vec (dot product)
        projection = circle_vec[0] * line_vec[0] + circle_vec[1] * line_vec[1]

        # Normalized projection (0 to 1 means closest point is on the segment)
        t = max(0, min(1, projection / line_len_sq)) if line_len_sq != 0 else 0

        # Closest point on the line segment to the circle center
        closest_point = (
            start[0] + t * line_vec[0],
            start[1] + t * line_vec[1]
        )

        # Distance from closest point to circle center
        distance_sq = (circle_center[0] - closest_point[0])**2 + \
                    (circle_center[1] - closest_point[1])**2

        return distance_sq <= radius**2

    def _update_passing_lanes(self):
        # Test cases
        assert self._line_intersects_circle((0,0), (10,0), (5,3), 3)  # True (line passes through circle)
        assert not self._line_intersects_circle((0,0), (10,0), (5,4), 3)  # False (line too far)
        assert self._line_intersects_circle((0,0), (0,0), (0,0), 1)  # True (zero-length segment at center)

        """Compute passing lanes for all players in both teams."""
        for team in [self.team1, self.team2]:
            opponents = self.team2 if team.controller == 1 else self.team1
            controlled_idx = max(0, team.control - 1) if team.control > 0 else 0
            controlled_player = team.players[controlled_idx] if team.control > 0 else team.goalie

            for i, player in enumerate(team.players):
                if i != controlled_idx:  # Skip controlled player
                    player.passing_lane_clear = self._is_passing_lane_clear(
                        (controlled_player.x, controlled_player.y),
                        (player.x, player.y),
                        opponents.players
                    )

    def _update_opponent_controlled_distances(self):
        """Update distances to the opponent's controlled player for all players."""
        for team in [self.team1, self.team2]:
            # Get the opponent's controlled player (goalie if control=0, else skater)
            opponent_team = self.team2 if team.controller == 1 else self.team1
            opp_controlled_player = (
                opponent_team.goalie if opponent_team.control == 0
                else opponent_team.players[opponent_team.control - 1]
            )

            # Update distances for all players in the current team
            for player in team.players + [team.goalie]:
                player.dist_to_controlled_opp = GameConsts.Distance(
                    (player.x, player.y),
                    (opp_controlled_player.x, opp_controlled_player.y)
                )

    def BeginFrame(self, info, action):
        self.action = action

        # Handle slapshot frames
        if action[5]:  # C button pressed
            if self.slapshot_frames_held == 0:
                self.slapshot_frames_held = 1
            else:
                self.slapshot_frames_held += 1
                if self.slapshot_frames_held >= self.SLAPSHOT_HOLD_FRAMES:
                    self.slapshot_frames_held = 0  # Reset after max hold
        else:
            self.slapshot_frames_held = 0  # Reset if C not pressed


        self.time = info.get("time")

        #Puck
        self.puck.x = info.get("puck_x")
        self.puck.y = info.get("puck_y")
        self.puck.vx = info.get("puck_vel_x")
        self.puck.vy = info.get("puck_vel_y")

        #Teams
        self.team1.begin_frame(info, self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)
        self.team2.begin_frame(info, self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)

        # Compute passing lanes AFTER both teams are updated
        self._update_passing_lanes()

        self._update_opponent_controlled_distances()

        #Normalize Puck
        self.nz_puck.x = self.puck.x / GameConsts.MAX_PUCK_X
        self.nz_puck.y = self.puck.y / GameConsts.MAX_PUCK_Y
        self.nz_puck.vx = self.puck.vx / GameConsts.MAX_VEL_XY
        self.nz_puck.vy = self.puck.vy / GameConsts.MAX_VEL_XY

    def EndFrame(self):
        self.last_time = self.time

        self.team1.end_frame()
        self.team2.end_frame()

        #self.debug_print()
        #time.sleep(5)

    def debug_print(self):
        print("\n" + "="*80)
        print(f"Game time: {self.time}, Last time: {self.last_time}")
        print(f"Slapshot frames held: {self.slapshot_frames_held}/{self.SLAPSHOT_HOLD_FRAMES}")

        # Puck info
        print("\nPuck State:")
        self.puck.debug_print("Raw Puck")
        self.nz_puck.debug_print("NZ Puck")

        # Team states
        print("\n" + "="*40 + " Team 1 State " + "="*40)
        self.team1.debug_print()

        print("\n" + "="*40 + " Team 2 State " + "="*40)
        self.team2.debug_print()

        # Action state
        action_names = ["Up", "Down", "Left", "Right", "B", "C"]
        print("\nCurrent Actions:")
        for name, val in zip(action_names, self.action):
            print(f"{name}: {'ON' if val else 'OFF'}", end=" | ")
        print()