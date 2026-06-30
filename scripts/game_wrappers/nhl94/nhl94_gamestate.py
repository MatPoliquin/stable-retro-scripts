"""
NHL94 Game State
"""

import math
from game_wrappers.nhl94.nhl94_const import GameConsts
from typing import Dict, Any
from copy import deepcopy
from dataclasses import dataclass
import time

@dataclass
class Net:
    depth: int = GameConsts.NET_DEPTH
    left: int = GameConsts.P1_NET_LEFT_POLL
    right: int = GameConsts.P1_NET_RIGHT_POLL
    y: int = 0 #center of net

    rel_controlled_left: int = 0
    rel_controlled_right: int = 0
    rel_controlled_y: int = 0 #center of net

@dataclass
class Player:
    x: int = 0
    y: int = 0
    vx: int = 0
    vy: int = 0
    anim: int = 0
    anim_frame: int = 0
    state_flags: int = 0
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
    one_timer_lane_good: bool = False
    is_one_timer: float = 0.0
    is_breakaway: float = 0.0
    is_falling: float = 0.0
    is_pad_stack: float = 0.0
    is_dive: float = 0.0

    def debug_print(self, prefix="Player"):
        print(f"{prefix} - x: {self.x}, y: {self.y}, vx: {self.vx}, vy: {self.vy}, "
            f"anim: {hex(self.anim)}, anim_frame: {self.anim_frame}, state_flags: {bin(self.state_flags)}, "
            f"orientation: {self.orientation}, ori_vec: ({self.ori_x:.2f}, {self.ori_y:.2f})\n"
            f"  rel_puck: ({self.rel_puck_x:.1f}, {self.rel_puck_y:.1f}) "
            f"rel_puck_vel: ({self.rel_puck_vx:.1f}, {self.rel_puck_vy:.1f})\n"
            f"  rel_controlled: ({self.rel_controlled_x:.1f}, {self.rel_controlled_y:.1f}) "
            f"rel_controlled_vel: ({self.rel_controlled_vx:.1f}, {self.rel_controlled_vy:.1f})\n"
            f"  dist_to_controlled: {self.dist_to_controlled:.1f}, "
            f"dist_to_controlled_opp: {self.dist_to_controlled_opp:.1f}, "
            f"dist_to_puck: {self.dist_to_puck:.1f}\n"
            f"  one_timer: {self.is_one_timer}, breakaway: {self.is_breakaway}, "
            f"falling: {self.is_falling}, pad_stack: {self.is_pad_stack}, dive: {self.is_dive}\n"
            f"  passing_lane_clear: {self.passing_lane_clear}, "
            f"one_timer_lane_good: {self.one_timer_lane_good}")

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


@dataclass
class EngineState:
    puck_owner: int = -1
    shot_player: int = -1
    pass_dir: int = 0
    pass_speed: int = 0
    goalie_chk_body: int = 0
    sflags: int = 0
    sflags2: int = 0
    ba_ps_flags: int = 0
    word_ffc2f6: int = 0
    word_ffc2f8: int = 0
    word_ffc2fa: int = 0
    shot_mode_active: float = 0.0
    shot_taken: float = 0.0
    in_close_top_shelf: float = 0.0
    one_timer_collision_mode: float = 0.0
    breakaway_context: float = 0.0
    controlled_is_shooter: float = 0.0
    goalie_box_small: float = 0.0

    def debug_print(self, prefix="EngineState"):
        print(
            f"{prefix} - puck_owner: {self.puck_owner}, shot_player: {self.shot_player}, "
            f"pass_dir: {self.pass_dir}, pass_speed: {self.pass_speed}, goalie_chk_body: {self.goalie_chk_body}\n"
            f"  shot_mode_active: {self.shot_mode_active}, shot_taken: {self.shot_taken}, "
            f"in_close_top_shelf: {self.in_close_top_shelf}, one_timer_collision_mode: {self.one_timer_collision_mode},\n"
            f"  breakaway_context: {self.breakaway_context}, controlled_is_shooter: {self.controlled_is_shooter}, "
            f"goalie_box_small: {self.goalie_box_small}"
        )

class Team():
    HAS_PUCK_TRESHOLD = 3
    PAD_STACK_ANIMS = {0x250, 0x2A2}
    DIVE_ANIM = 0x2F4

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

        self.net = Net()
        self.nz_net = Net()

        net_left: tuple = (0.0, 0.0)        # Normalized absolute left post
        nz_net_right: tuple = (0.0, 0.0)       # Normalized absolute right post
        nz_net_center: tuple = (0.0, 0.0)      # Normalized absolute center
        nz_net_left_rel: tuple = (0.0, 0.0)    # Relative to controlled player
        nz_net_right_rel: tuple = (0.0, 0.0)
        nz_net_center_rel: tuple = (0.0, 0.0)

    def get_controlled_player(self) -> Player:
        return self.goalie if self.control == 0 else self.players[self.control - 1]

    def controlled_scnum(self) -> int:
        if self.control == 0:
            return 5 if self.controller == 1 else 11
        base_scnum = 0 if self.controller == 1 else 6
        return base_scnum + (self.control - 1)

    def skater_scnum_base(self) -> int:
        return 0 if self.controller == 1 else 6

    def goalie_scnum(self) -> int:
        return 5 if self.controller == 1 else 11

    def owns_scnum(self, scnum: int) -> bool:
        skater_base = self.skater_scnum_base()
        return skater_base <= scnum < (skater_base + self.num_players) or scnum == self.goalie_scnum()

    def get_player_by_scnum(self, scnum: int) -> Player | None:
        if scnum == self.goalie_scnum():
            return self.goalie

        skater_base = self.skater_scnum_base()
        skater_index = scnum - skater_base
        if 0 <= skater_index < self.num_players:
            return self.players[skater_index]

        return None

    def get_possession_player(self) -> Player | None:
        if self.goalie_haspuck:
            return self.goalie

        for player in self.players:
            if self.has_puck(player.x, player.y):
                return player

        return None

    def _load_hidden_player_fields(self, player: Player, info: Dict[str, Any], prefix: str) -> None:
        player.anim = info.get(f"{prefix}anim", 0) or 0
        player.anim_frame = info.get(f"{prefix}anim_frame", 0) or 0
        player.state_flags = info.get(f"{prefix}state_flags", 0) or 0

        player.is_one_timer = 1.0 if player.state_flags & (1 << 3) else 0.0
        player.is_breakaway = 1.0 if player.state_flags & (1 << 1) else 0.0
        player.is_falling = 1.0 if player.state_flags & (1 << 5) else 0.0
        player.is_pad_stack = 1.0 if player.anim in self.PAD_STACK_ANIMS else 0.0
        player.is_dive = 1.0 if player.anim == self.DIVE_ANIM else 0.0

        nz_net_left: tuple = (0.0, 0.0)        # Normalized absolute left post
        nz_net_right: tuple = (0.0, 0.0)       # Normalized absolute right post
        nz_net_center: tuple = (0.0, 0.0)      # Normalized absolute center
        nz_net_left_rel: tuple = (0.0, 0.0)    # Relative to controlled player
        nz_net_right_rel: tuple = (0.0, 0.0)
        nz_net_center_rel: tuple = (0.0, 0.0)

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
            self.stats.fullstar_x = info.get("fullstar_x")
            self.stats.fullstar_y = info.get("fullstar_y")
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
        self._load_hidden_player_fields(self.goalie, info, f"{self.ram_var_goalie_prefix}")

        # Players
        for p in range(0, self.num_players):
            if p == 0:
                self.players[p].x = info.get(f"{self.ram_var_prefix}x")
                self.players[p].y = info.get(f"{self.ram_var_prefix}y")
                self.players[p].vx = info.get(f"{self.ram_var_prefix}vel_x")
                self.players[p].vy = info.get(f"{self.ram_var_prefix}vel_y")
                self.players[p].orientation = info.get(f"{self.ram_var_prefix}ori", 0)
                self._load_hidden_player_fields(self.players[p], info, self.ram_var_prefix)
            else:
                pi = p + 1
                self.players[p].x = info.get(f"{self.ram_var_prefix}{pi}_x")
                self.players[p].y = info.get(f"{self.ram_var_prefix}{pi}_y")
                self.players[p].vx = info.get(f"{self.ram_var_prefix}{pi}_vel_x")
                self.players[p].vy = info.get(f"{self.ram_var_prefix}{pi}_vel_y")
                self.players[p].orientation = info.get(f"{self.ram_var_prefix}{pi}_ori", 0)
                self._load_hidden_player_fields(self.players[p], info, f"{self.ram_var_prefix}{pi}_")

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

    def Normalize(self):
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

        # Normalize
        for p in range(0, self.num_players):
            self.nz_players[p].dist_to_controlled_opp = self.players[p].dist_to_controlled_opp / GameConsts.MAX_PLAYER_X
            self.nz_players[p].passing_lane_clear = float(self.players[p].passing_lane_clear)  # Convert bool to 0.0/1.0
            self.nz_players[p].one_timer_lane_good = float(self.players[p].one_timer_lane_good)

        # Normalize net positions
        # Absolute positions
        self.nz_net.left = self.net.left / GameConsts.MAX_PUCK_X
        self.nz_net.right = self.net.right / GameConsts.MAX_PUCK_X
        self.nz_net.y = self.net.y / GameConsts.MAX_PUCK_Y
        self.nz_net.depth = self.net.depth / GameConsts.MAX_PUCK_Y

        # Relative positions
        self.nz_net.rel_controlled_left = self.net.rel_controlled_left / GameConsts.MAX_PUCK_X
        self.nz_net.rel_controlled_right = self.net.rel_controlled_right / GameConsts.MAX_PUCK_X
        self.nz_net.rel_controlled_y = self.net.rel_controlled_y / GameConsts.MAX_PUCK_Y

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
        self.engine = EngineState()
        self.period = 1
        self.time = 0
        self.last_time = 0
        self.numPlayers = numPlayers

        self.action = [0] * 6 # Up, Down, Left, Right, B, C

        # Slapshot tracking
        self.slapshot_frames_held = 0
        self.SLAPSHOT_HOLD_FRAMES = 60  # Max frames to hold for slapshot

        #For model input
        self.nz_puck = Player()

    def _update_engine_state(self, info: Dict[str, Any]) -> None:
        puck_owner = info.get("puck_owner")
        shot_player = info.get("shot_player")

        self.engine.puck_owner = -1 if puck_owner is None else puck_owner
        self.engine.shot_player = -1 if shot_player is None else shot_player
        self.engine.pass_dir = info.get("pass_dir", 0) or 0
        self.engine.pass_speed = info.get("pass_speed", 0) or 0
        self.engine.goalie_chk_body = info.get("goalie_chk_body", 0) or 0
        self.engine.sflags = info.get("sflags", 0) or 0
        self.engine.sflags2 = info.get("sflags2", 0) or 0
        self.engine.ba_ps_flags = info.get("ba_ps_flags", 0) or 0
        self.engine.word_ffc2f6 = info.get("word_ffc2f6", 0) or 0
        self.engine.word_ffc2f8 = info.get("word_ffc2f8", 0) or 0
        self.engine.word_ffc2fa = info.get("word_ffc2fa", 0) or 0

        self.engine.shot_mode_active = 1.0 if self.engine.sflags & (1 << 3) else 0.0
        self.engine.shot_taken = 1.0 if self.engine.sflags2 & (1 << 4) else 0.0
        self.engine.in_close_top_shelf = 1.0 if self.engine.word_ffc2f6 & (1 << 4) else 0.0
        self.engine.one_timer_collision_mode = 1.0 if self.engine.word_ffc2f8 & (1 << 1) else 0.0
        self.engine.breakaway_context = 1.0 if self.engine.word_ffc2fa & (1 << 4) else 0.0
        self.engine.controlled_is_shooter = 1.0 if self.engine.shot_player == self.team1.controlled_scnum() else 0.0
        self.engine.goalie_box_small = 1.0 if self.engine.goalie_chk_body <= 0xC else 0.0

    # Flip the variables
    def Flip(self):
        self.team1, self.team2 = self.team2, self.team1
        return

    @staticmethod
    def _team_attacking_top(team: Team) -> bool:
        return team.controller == 1

    def _normalized_attack_y(self, player: Player, team: Team) -> int:
        return player.y if self._team_attacking_top(team) else -player.y

    def _is_receiver_in_attack_band(self, player: Player, team: Team) -> bool:
        normalized_y = self._normalized_attack_y(player, team)
        return GameConsts.ATACKZONE_POS_Y <= normalized_y <= GameConsts.P2_NET_Y

    @staticmethod
    def _is_receiver_in_central_lane(player: Player) -> bool:
        return GameConsts.SLOT_BAND_MIN_X <= player.x <= GameConsts.SLOT_BAND_MAX_X

    def _get_passer_for_team(self, team: Team) -> Player | None:
        puck_owner = self.engine.puck_owner
        if team.owns_scnum(puck_owner):
            return team.get_player_by_scnum(puck_owner)

        return team.get_possession_player()

    def _is_goalie_player(self, player: Player) -> bool:
        return player is self.team1.goalie or player is self.team2.goalie

    def _passing_intercept_radius(self, player: Player) -> int:
        if not self._is_goalie_player(player):
            return 14

        if player.is_pad_stack:
            return 18

        goalie_radius = self.engine.goalie_chk_body if self.engine.goalie_chk_body > 0 else 12
        return max(12, min(goalie_radius, 18))

    @staticmethod
    def _receiver_catch_point(player: Player) -> tuple[int, int]:
        catch_offset = 10
        return (
            player.x + round(player.ori_x * catch_offset),
            player.y + round(player.ori_y * catch_offset),
        )

    def _attacking_net_for_team(self, team: Team) -> Net:
        return self.team2.net if self._team_attacking_top(team) else self.team1.net

    @staticmethod
    def _shot_target_points_for_net(net: Net) -> list[tuple[int, int]]:
        center_x = (net.left + net.right) // 2
        post_inset = 6
        return [
            (center_x, net.y),
            (net.left + post_inset, net.y),
            (net.right - post_inset, net.y),
        ]

    def _get_clear_one_timer_shot_target(self, shooter: Player, team: Team, opponents: Team) -> tuple[int, int] | None:
        shot_start = (shooter.x, shooter.y)
        attacking_net = self._attacking_net_for_team(team)

        for target_point in self._shot_target_points_for_net(attacking_net):
            if self._is_passing_lane_clear(shot_start, target_point, opponents.players):
                return target_point

        return None

    def _has_clear_one_timer_shot_lane(self, shooter: Player, team: Team, opponents: Team) -> bool:
        return self._get_clear_one_timer_shot_target(shooter, team, opponents) is not None

    def _pass_direction_matches_target(self, passer: Player, target_point: tuple[int, int]) -> bool:
        if not 0 <= self.engine.pass_dir <= 7:
            return True

        delta_x = target_point[0] - passer.x
        delta_y = target_point[1] - passer.y
        distance = math.hypot(delta_x, delta_y)
        if distance == 0:
            return False

        target_x = delta_x / distance
        target_y = delta_y / distance
        pass_angle = self.engine.pass_dir * (2 * math.pi / 8)
        pass_dir_x = math.cos(pass_angle)
        pass_dir_y = math.sin(pass_angle)
        dot_product = (target_x * pass_dir_x) + (target_y * pass_dir_y)
        return dot_product >= math.cos(math.pi / 4)

    def _is_passing_lane_clear(self, start_pos, end_pos, opponents):
        """
        Check if a straight-line path between two points is obstructed.
        
        Based on NHL94 ROM data:
        - Stick interception: 14 units from stick hotspot (dist^2 <= 196)
        - Body collision: 8 units from player center (dist^2 <= 64)
        - Broad-phase Y-filter: 22 units (ROM optimization)
        
        Using radius=18 (14 + 4 buffer for moving stick hotspot)
        """
        for opponent in opponents:
            radius = self._passing_intercept_radius(opponent)

            # Broad-phase: skip if Y difference is too large (ROM optimization)
            min_y_dist = min(abs(opponent.y - start_pos[1]), abs(opponent.y - end_pos[1]))
            if min_y_dist > radius + 8:
                continue
                
            if self._line_intersects_circle(start_pos, end_pos, (opponent.x, opponent.y), radius=radius):
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

    def _line_intersects_line(self, line1_start, line1_end, line2_start, line2_end):
        """Check if two line segments intersect.

        Args:
            line1_start: Tuple (x, y) of first line's start point
            line1_end: Tuple (x, y) of first line's end point
            line2_start: Tuple (x, y) of second line's start point
            line2_end: Tuple (x, y) of second line's end point

        Returns:
            bool: True if the line segments intersect
        """
        # Implementation of the line segment intersection algorithm
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        A = line1_start
        B = line1_end
        C = line2_start
        D = line2_end

        # Check if lines intersect
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def _update_passing_lanes(self):
        """Compute pass lanes and one-timer lanes for both teams."""
        for team in [self.team1, self.team2]:
            for player in team.players:
                player.passing_lane_clear = False
                player.one_timer_lane_good = False

            opponents = self.team2 if team.controller == 1 else self.team1
            passer = self._get_passer_for_team(team)
            if passer is None:
                continue

            passer_in_attack_band = self._is_receiver_in_attack_band(passer, team)

            # Get all potential obstacles
            obstacles = []

            # 1. Add opponent players
            obstacles.extend(opponents.players)

            # 2. Add opponent goalie
            if passer is not opponents.goalie:
                obstacles.append(opponents.goalie)

            for i, player in enumerate(team.players):
                if player is not passer:
                    catch_point = self._receiver_catch_point(player)
                    path_clear = self._pass_direction_matches_target(passer, catch_point)
                    if path_clear:
                        path_clear = self._is_passing_lane_clear(
                            (passer.x, passer.y),
                            catch_point,
                            obstacles,
                        )

                    in_attack_band = self._is_receiver_in_attack_band(player, team)
                    in_central_lane = self._is_receiver_in_central_lane(player)
                    shot_lane_clear = False
                    if path_clear and in_attack_band and in_central_lane:
                        shot_lane_clear = self._has_clear_one_timer_shot_lane(player, team, opponents)

                    player.passing_lane_clear = path_clear
                    player.one_timer_lane_good = (
                        path_clear
                        and passer_in_attack_band
                        and in_attack_band
                        and in_central_lane
                        and shot_lane_clear
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

    def update_nets(self):
        # Update team 1 net (bottom of screen)
        self.team1.net.y = GameConsts.P1_NET_Y
        self.team1.net.left = GameConsts.P1_NET_LEFT_POLL
        self.team1.net.right = GameConsts.P1_NET_RIGHT_POLL

        # Update team 2 net (top of screen)
        self.team2.net.y = GameConsts.P2_NET_Y
        self.team2.net.left = GameConsts.P2_NET_LEFT_POLL
        self.team2.net.right = GameConsts.P2_NET_RIGHT_POLL

        # Calculate relative net positions for both teams
        for team in [self.team1, self.team2]:
            # Get controlled player position
            controlled_x = team.goalie.x if team.control == 0 else team.players[team.control-1].x
            controlled_y = team.goalie.y if team.control == 0 else team.players[team.control-1].y

            # Calculate relative net positions
            team.net.rel_controlled_left = team.net.left - controlled_x
            team.net.rel_controlled_right = team.net.right - controlled_x
            team.net.rel_controlled_y = team.net.y - controlled_y



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


        self.period = info.get("period", 1) or 1
        self.time = info.get("time")

        #Puck
        self.puck.x = info.get("puck_x")
        self.puck.y = info.get("puck_y")
        self.puck.vx = info.get("puck_vel_x")
        self.puck.vy = info.get("puck_vel_y")

        #Teams
        self.team1.begin_frame(info, self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)
        self.team2.begin_frame(info, self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)
        self._update_engine_state(info)

        self.update_nets()

        # Compute passing lanes AFTER both teams are updated
        self._update_passing_lanes()

        self._update_opponent_controlled_distances()

        self.team1.Normalize()
        self.team2.Normalize()

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
        self.engine.debug_print()