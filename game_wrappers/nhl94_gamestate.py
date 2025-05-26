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

@dataclass
class Stats:
    score: int = 0
    shots: int = 0
    bodychecks: int = 0
    attackzone: int = 0
    faceoffwon: int = 0
    passing: int = 0
    fullstar_x: int = 0
    fullstar_y: int = 0
    emptystar_x: int = 0
    emptystar_y: int = 0

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
        #print(pos_x, pos_y, self.stats.fullstar_x, self.stats.fullstar_y)
        return (abs(pos_x - self.stats.fullstar_x) < self.HAS_PUCK_TRESHOLD and abs(pos_y - self.stats.fullstar_y) < self.HAS_PUCK_TRESHOLD)

    def has_control(self, pos_x, pos_y):
        return (abs(pos_x - self.stats.emptystar_x) < self.HAS_PUCK_TRESHOLD and abs(pos_y - self.stats.emptystar_y) < self.HAS_PUCK_TRESHOLD)

    def begin_frame(self, info: Dict[str, Any]) -> None:
        # General state
        self.stats.score = info.get(f"{self.ram_var_prefix}score")
        self.stats.shots = info.get(f"{self.ram_var_prefix}shots")
        self.stats.bodychecks = info.get(f"{self.ram_var_prefix}bodychecks")
        self.stats.attackzone = info.get(f"{self.ram_var_prefix}attackzone")
        self.stats.faceoffwon = info.get(f"{self.ram_var_prefix}faceoffwon")
        self.stats.passing = info.get(f"{self.ram_var_prefix}passing")
        self.stats.fullstar_x = info.get(f"{self.ram_var_prefix}fullstar_x", 0)
        self.stats.fullstar_y = info.get(f"{self.ram_var_prefix}fullstar_y", 0)
        self.stats.emptystar_x = info.get(f"{self.ram_var_prefix}emptystar_x", 0)
        self.stats.emptystar_y = info.get(f"{self.ram_var_prefix}emptystar_y", 0)

        # Goalie
        self.goalie.x = info.get(f"{self.ram_var_goalie_prefix}x")
        self.goalie.y = info.get(f"{self.ram_var_goalie_prefix}y")
        self.goalie.vx = info.get(f"{self.ram_var_goalie_prefix}vel_x", 0)
        self.goalie.vy = info.get(f"{self.ram_var_goalie_prefix}vel_y", 0)

        # Players
        for p in range(0, self.num_players - 1):
            if p == 0:
                self.players[p].x = info.get(f"{self.ram_var_prefix}x")
                self.players[p].y = info.get(f"{self.ram_var_prefix}y")
                self.players[p].vx = info.get(f"{self.ram_var_prefix}vel_x")
                self.players[p].vy = info.get(f"{self.ram_var_prefix}vel_y")
            else:
                pi = p + 1
                self.players[p].x = info.get(f"{self.ram_var_prefix}{pi}_x")
                self.players[p].y = info.get(f"{self.ram_var_prefix}{pi}_y")
                self.players[p].vx = info.get(f"{self.ram_var_prefix}{pi}_vel_x")
                self.players[p].vy = info.get(f"{self.ram_var_prefix}{pi}_vel_y")



        # Knowing if the player has the puck is tricky since the fullstar in the game is not aligned with the player every frame
        # There is an offset of up to 2 sometimes
        self.player_haspuck = False
        for p in range(0, self.num_players - 1):
            if self.has_puck(self.players[p].x, self.players[p].y):
                self.player_haspuck = True

        self.goalie_haspuck = self.has_puck(self.goalie.x, self.goalie.y)

        # Check which player is beeing controlled
        if self.goalie_haspuck:
            self.control = 0
        elif self.player_haspuck:
            for p in range(0, self.num_players - 1):
                if self.has_puck(self.players[p].x, self.players[p].y):
                    self.control = p + 1
        else:
            for p in range(0, self.num_players - 1):
                if self.has_control(self.players[p].x, self.players[p].y):
                    self.control = p + 1

    def end_frame(self) -> None:
        self.last_stats = deepcopy(self.stats)
        self.last_distToPuck = self.distToPuck

        # Normalize for model input. TODO: also make positions and velocities relative to controlled player
        for p in range(0, self.num_players - 1):
            self.nz_players[p].x = self.players[p].x / GameConsts.MAX_PLAYER_X
            self.nz_players[p].y = self.players[p].y / GameConsts.MAX_PLAYER_Y
            self.nz_players[p].vx = self.players[p].vx / GameConsts.MAX_VEL_XY
            self.nz_players[p].vy = self.players[p].vy / GameConsts.MAX_VEL_XY

        self.nz_goalie.x = self.goalie.x / GameConsts.MAX_PLAYER_X
        self.nz_goalie.y = self.goalie.y / GameConsts.MAX_PLAYER_Y
        self.nz_goalie.vx = self.goalie.vx / GameConsts.MAX_VEL_XY
        self.nz_goalie.vy = self.goalie.vy / GameConsts.MAX_VEL_XY

        self.nz_player_haspuck = 1.0 if self.player_haspuck else 0.0
        self.nz_goalie_haspuck = 1.0 if self.goalie_haspuck else 0.0


class NHL94GameState():
    def __init__(self, numPlayers):
        self.team1 = Team(0, numPlayers)
        self.team2 = Team(1, numPlayers)
        self.puck = Player()
        self.time = 0
        self.last_time = 0

        #For model input
        self.nz_puck = Player()

    # Flip the variables
    def Flip(self):
        self.team1, self.team2 = self.team2, self.team1

    def BeginFrame(self, info):
        self.time = info.get("time")

        #Puck
        self.puck.x = info.get("puck_x")
        self.puck.y = info.get("puck_y")
        self.puck.vx = info.get("puck_vel_x")
        self.puck.vy = info.get("puck_vel_y")

        self.team1.begin_frame(info)
        self.team2.begin_frame(info)

        # Distance
        self.team1.distToPuck = GameConsts.Distance((self.team1.players[0].x, self.team1.players[0].y), (self.puck.x, self.puck.y))
        self.team2.distToPuck = GameConsts.Distance((self.team2.players[0].x, self.team2.players[0].y), (self.puck.x, self.puck.y))

    def EndFrame(self):
        self.last_time = self.time

        #Puck
        self.nz_puck.x = self.puck.x / GameConsts.MAX_PUCK_X
        self.nz_puck.y = self.puck.y / GameConsts.MAX_PUCK_Y
        self.nz_puck.vx = self.puck.vx / GameConsts.MAX_VEL_XY
        self.nz_puck.vy = self.puck.vy / GameConsts.MAX_VEL_XY

        self.team1.end_frame()
        self.team2.end_frame()
