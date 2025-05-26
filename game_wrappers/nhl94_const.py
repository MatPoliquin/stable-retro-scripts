"""
Game Constants
"""

import math

class GameConsts():
    INPUT_B = 0
    INPUT_A = 1
    INPUT_MODE = 2
    INPUT_START = 3
    INPUT_UP = 4
    INPUT_DOWN = 5
    INPUT_LEFT = 6
    INPUT_RIGHT = 7
    INPUT_C = 8
    INPUT_Y = 9
    INPUT_X = 10
    INPUT_Z = 11
    INPUT_MAX = 12

    P1_NET_Y = -264
    P1_NET_LEFT_POLL = -19
    P1_NET_RIGHT_POLL = 19

    P2_NET_Y = 264
    P2_NET_LEFT_POLL = -19
    P2_NET_RIGHT_POLL = 19

    SHOOT_POS_X = 0
    SHOOT_POS_Y = 120

    ATACKZONE_POS_Y = 100
    DEFENSEZONE_POS_Y = -80

    MAX_PLAYER_X = 120
    MAX_PLAYER_Y = 270

    MAX_PUCK_X = 130
    MAX_PUCK_Y = 270

    MAX_VEL_XY = 50

    # Used for Reward Functions
    SPAWNABLE_AREA_WIDTH = 235
    SPAWNABLE_AREA_HEIGHT = 460
    SPAWNABLE_ZONE_HEIGHT = 140

    CREASE_UPPER_BOUND = 230
    CREASE_LOWER_BOUND = 210
    CREASE_MIN_VEL = 30
    CREASE_MAX_X = 23
    CREASE_MIN_GOALIE_PUCK_DIST_X = 7

    def Distance(vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
        return math.sqrt(tmp)

    def DistToPos(vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
        return math.sqrt(tmp)