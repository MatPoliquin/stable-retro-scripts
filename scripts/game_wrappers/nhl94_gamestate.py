"""
NHL94 Game State
"""

import math
from game_wrappers.nhl94_const import GameConsts
from typing import Dict, Any
from copy import deepcopy
from dataclasses import dataclass

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

    def debug_print(self, prefix="Player"):
        print(f"{prefix} - x: {self.x}, y: {self.y}, vx: {self.vx}, vy: {self.vy}, "
              f"orientation: {self.orientation}, ori_vec: ({self.ori_x:.2f}, {self.ori_y:.2f})")

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
        self.distToPuck = 0
        self.last_distToPuck = 0

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

        # 0.0 and 1.0 switched around due to current models trained that way
        self.nz_player_haspuck = 0.0 if self.player_haspuck else 1.0
        self.nz_goalie_haspuck = 0.0 if self.goalie_haspuck else 1.0

    def end_frame(self) -> None:
        self.last_stats = deepcopy(self.stats)
        self.last_distToPuck = self.distToPuck

    def debug_print(self):
        print(f"Team controller: {self.controller}")
        print(f"Team player prefix: {self.ram_var_prefix}")
        self.stats.debug_print("Stats")
        self.last_stats.debug_print("Last Stats")
        print(f"Number of players: {self.num_players}")
        for idx, player in enumerate(self.players):
            player.debug_print(f"Player {idx}")
        self.goalie.debug_print("Goalie")
        print(f"Control: {self.control}")
        print(f"Player has puck: {self.player_haspuck}")
        print(f"Goalie has puck: {self.goalie_haspuck}")
        print(f"Distance to puck: {self.distToPuck}, Last distance to puck: {self.last_distToPuck}")
        print("Normalized players:")
        for idx, player in enumerate(self.nz_players):
            player.debug_print(f"NZ Player {idx}")
        self.nz_goalie.debug_print("NZ Goalie")
        print(f"NZ Player has puck: {self.nz_player_haspuck}")
        print(f"NZ Goalie has puck: {self.nz_goalie_haspuck}")


class NHL94GameState():
    def __init__(self, numPlayers):
        self.team1 = Team(1, numPlayers)
        self.team2 = Team(2, numPlayers)
        self.puck = Player()
        self.time = 0
        self.last_time = 0
        self.numPlayers = numPlayers

        #For model input
        self.nz_puck = Player()

    # Flip the variables
    def Flip(self):
        self.team1, self.team2 = self.team2, self.team1
        return

    def BeginFrame(self, info):
        self.time = info.get("time")

        #Puck
        self.puck.x = info.get("puck_x")
        self.puck.y = info.get("puck_y")
        self.puck.vx = info.get("puck_vel_x")
        self.puck.vy = info.get("puck_vel_y")

        self.team1.begin_frame(info, self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)
        self.team2.begin_frame(info, self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)

        # Distance
        self.team1.distToPuck = GameConsts.Distance((self.team1.players[0].x, self.team1.players[0].y), (self.puck.x, self.puck.y))
        self.team2.distToPuck = GameConsts.Distance((self.team2.players[0].x, self.team2.players[0].y), (self.puck.x, self.puck.y))

        #Puck
        self.nz_puck.x = self.puck.x / GameConsts.MAX_PUCK_X
        self.nz_puck.y = self.puck.y / GameConsts.MAX_PUCK_Y
        self.nz_puck.vx = self.puck.vx / GameConsts.MAX_VEL_XY
        self.nz_puck.vy = self.puck.vy / GameConsts.MAX_VEL_XY

    def EndFrame(self):
        self.last_time = self.time

        self.team1.end_frame()
        self.team2.end_frame()

        #self.debug_print()

    def debug_print(self):
        print("===================================================================")
        print(f"Game time: {self.time}, Last time: {self.last_time}")
        self.puck.debug_print("Puck")
        self.nz_puck.debug_print("NZ Puck")
        print("=== Team 1 State: ===")
        self.team1.debug_print()
        print("=== Team 2 State: ===")
        self.team2.debug_print()