"""
NHL94 Game State
"""

import math
from game_wrappers.nhl94_const import GameConsts

class NHL94GameState():
    def __init__(self):
        self.p1_score = 0
        self.p2_score = 0
        self.p1_shots = 0
        self.p2_shots = 0
        self.p1_bodychecks = 0
        self.p2_attackzone = 0
        self.p1_attackzone = 0
        self.p1_faceoffwon = 0
        self.p1_passing = 0
        self.last_p1_score = 0
        self.last_p2_score = 0
        self.last_p1_shots = 0
        self.last_p1_bodychecks = 0
        self.last_p1_attackzone = 0
        self.last_p2_attackzone = 0
        self.last_p1_faceoffwon = 0
        self.last_p2_shots = 0
        self.last_p1_pos = (0,0)
        self.last_p2_pos = (0,0)
        self.last_puck_pos = (0,0)
        self.last_time = 0
        self.last_p1_passing = 0
        self.p1_x = 0
        self.p1_y = 0
        self.p1_2_x = 0
        self.p1_2_y = 0
        self.p2_x = 0
        self.p2_y = 0
        self.p2_2_x = 0
        self.p2_2_y = 0
        self.g1_x = 0
        self.g1_y = 0
        self.g2_x = 0
        self.g2_y = 0
        self.last_dist = -1
        self.last_dist_az = -1
        self.counter = 0
        self.lastshot_time = -1
        self.time = 0
        self.last_havepuck_time = -1
        self.distToPuck = 0
        self.p1_emptystar_x = 0
        self.p1_emptystar_y = 0
        self.p1_fullstar_x = 0
        self.p1_fullstar_y = 0
        self.p2_fullstar_x = 0
        self.p2_fullstar_y = 0
        self.player_haspuck = False
        self.goalie_haspuck = False
        self.p2_haspuck = False
        self.g2_haspuck = False
        self.p1_vel_x  = 0
        self.p1_vel_y  = 0
        self.p1_2_vel_x  = 0
        self.p1_2_vel_y  = 0
        self.p2_vel_x  = 0
        self.p2_vel_y  = 0
        self.p2_2_vel_x  = 0
        self.p2_2_vel_y  = 0
        self.puck_vel_x  = 0
        self.puck_vel_y  = 0
        self.puck_x = 0
        self.puck_y = 0

        self.normalized_p1_x = 0
        self.normalized_p1_y = 0
        self.normalized_p1_2_x = 0
        self.normalized_p1_2_y = 0
        self.normalized_p2_x = 0
        self.normalized_p2_y = 0
        self.normalized_p2_2_x = 0
        self.normalized_p2_2_y = 0
        self.normalized_g1_x = 0
        self.normalized_g1_y = 0
        self.normalized_g2_x = 0
        self.normalized_g2_y = 0
        self.normalized_puck_x = 0
        self.normalized_puck_y = 0
        self.normalized_puck_vel_x = 0
        self.normalized_puck_vel_y = 0
        self.normalized_player_haspuck = 0.0
        self.normalized_goalie_haspuck = 0.0

        self.normalized_p1_velx = 0.0
        self.normalized_p1_vely = 0.0
        self.normalized_p1_2_velx = 0.0
        self.normalized_p1_2_vely = 0.0
        self.normalized_p2_velx = 0.0
        self.normalized_p2_vely = 0.0
        self.normalized_p2_2_velx = 0.0
        self.normalized_p2_2_vely = 0.0

        self.p1_control = 0 # 0 = g1, 1 = p1_1, 2 = p1_2


    def swap(self, x,y):
        tmp = x
        x = y
        y = tmp

    # Flip the variables so p2 becomes p1
    def Flip(self):
        self.p1_x, self.p2_x = self.p2_x, self.p1_x
        self.p1_y, self.p2_y = self.p2_y, self.p1_y
        self.g1_x, self.g2_x = self.g2_x, self.g1_x
        self.g1_y, self.g2_y = self.g2_y, self.g1_y
        self.p1_fullstar_x, self.p2_fullstar_x = self.p2_fullstar_x, self.p1_fullstar_x
        self.p1_fullstar_y, self.p2_fullstar_y = self.p2_fullstar_y, self.p1_fullstar_y

        self.player_haspuck, self.p2_haspuck = self.p2_haspuck, self.player_haspuck
        self.goalie_haspuck, self.g2_haspuck = self.g2_haspuck, self.goalie_haspuck

        #print('AFTER:%d,%d\n' % (self.p1_x, self.p2_x))

    def Distance(self, vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
    
        return math.sqrt(tmp)

    def DistToPos(self, vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
    
        return math.sqrt(tmp)
    
    def has_puck(self, pos_x, pos_y):
        return (abs(pos_x - self.p1_fullstar_x) < 3 and abs(pos_y - self.p1_fullstar_y) < 3)
    
    def has_control(self, pos_x, pos_y):
        return (abs(pos_x - self.p1_emptystar_x) < 3 and abs(pos_y - self.p1_emptystar_y) < 3)


    def BeginFrame(self, info):
        self.p1_score = info.get('p1_score')
        self.p2_score = info.get('p2_score')
        self.p1_shots = info.get('p1_shots')
        self.p2_shots = info.get('p2_shots')
        self.p1_bodychecks = info.get('p1_bodychecks')
        self.p2_attackzone = info.get('p2_attackzone')
        self.p1_attackzone = info.get('p1_attackzone')
        self.p1_faceoffwon = info.get('p1_faceoffwon')
        self.p1_passing = info.get('p1_passing')
        self.p1_x = info.get('p1_x')
        self.p1_y = info.get('p1_y')
        self.p1_2_x = info.get('p1_2_x')
        self.p1_2_y = info.get('p1_2_y')
        self.p2_x = info.get('p2_x')
        self.p2_y = info.get('p2_y')
        self.p2_2_x = info.get('p2_2_x')
        self.p2_2_y = info.get('p2_2_y')
        self.g1_x = info.get('g1_x')
        self.g1_y = info.get('g1_y')
        self.g2_x = info.get('g2_x')
        self.g2_y = info.get('g2_y')
        self.time = info.get('time')
        self.puck_x = info.get('puck_x')
        self.puck_y = info.get('puck_y')
        self.puck_vel_x = info.get('puck_vel_x')
        self.puck_vel_y = info.get('puck_vel_y')
        self.p1_vel_x = info.get('p1_vel_x')
        self.p1_vel_y = info.get('p1_vel_y')
        self.p1_2_vel_x = info.get('p1_2_vel_x')
        self.p1_2_vel_y = info.get('p1_2_vel_y')
        self.p2_vel_x = info.get('p2_vel_x')
        self.p2_vel_y = info.get('p2_vel_y')
        self.p2_2_vel_x = info.get('p2_2_vel_x')
        self.p2_2_vel_y = info.get('p2_2_vel_y')
        self.p1_fullstar_x = info.get('fullstar_x')
        self.p1_fullstar_y = info.get('fullstar_y')
        self.p2_fullstar_x = info.get('p2_fullstar_x')
        self.p2_fullstar_y = info.get('p2_fullstar_y')
        self.p1_emptystar_x = info.get('p1_emptystar_x')
        self.p1_emptystar_y = info.get('p1_emptystar_y')

        self.distToPuck = self.Distance((self.p1_x, self.p1_y), (self.puck_x, self.puck_y))


        # Knowing if the player has the puck is tricky since the fullstar in the game is not aligned with the player every frame
        # There is an offset of up to 2 sometimes

        self.player_haspuck = self.has_puck(self.p1_x, self.p1_y) or self.has_puck(self.p1_2_x, self.p1_2_y)
        self.p2_haspuck = self.has_puck(self.p2_x, self.p2_y)

        self.goalie_haspuck = self.has_puck(self.g1_x, self.g1_y)
        self.g2_haspuck = self.has_puck(self.g2_x, self.g2_y)

        # TODO check which p1 player is beeing controlled
        #self.p1_control

        if self.goalie_haspuck:
            self.p1_control = 0
        elif self.player_haspuck:
            if self.has_puck(self.p1_x, self.p1_y):
                self.p1_control = 1
            elif self.has_puck(self.p1_2_x, self.p1_2_y):
                self.p1_control = 2
        else:
            if self.has_control(self.p1_x, self.p1_y):
                self.p1_control = 1
            elif self.has_control(self.p1_2_x, self.p1_2_y):
                self.p1_control = 2

    def EndFrame(self):
        self.counter += 1
        if self.counter == 10:
            self.last_p1_pos = (self.p1_x, self.p1_y)
            self.last_p2_pos = (self.p2_x, self.p2_x)
            self.last_puck_pos = (self.puck_x, self.puck_y)
            self.counter = 0

        self.last_p1_score = self.p1_score
        self.last_p1_shots = self.p1_shots
        self.last_p1_bodychecks = self.p1_bodychecks
        self.last_p2_attackzone = self.p2_attackzone
        self.last_p1_attackzone = self.p1_attackzone
        self.last_p1_faceoffwon = self.p1_faceoffwon
        self.last_p2_shots = self.p2_shots
        self.last_p2_score = self.p2_score
        self.last_time = self.time
        self.last_p1_passing = self.p1_passing
        self.last_dist = self.distToPuck


        self.normalized_p1_x = self.p1_x / GameConsts.MAX_PLAYER_X
        self.normalized_p1_y = self.p1_y / GameConsts.MAX_PLAYER_X
        self.normalized_p1_2_x = self.p1_2_x / GameConsts.MAX_PLAYER_X
        self.normalized_p1_2_y = self.p1_2_y / GameConsts.MAX_PLAYER_Y
        self.normalized_p2_x = (self.p2_x - self.p1_x)/ GameConsts.MAX_PLAYER_X
        self.normalized_p2_y = (self.p2_y - self.p1_y)/ GameConsts.MAX_PLAYER_Y
        self.normalized_p2_2_x = (self.p2_2_x - self.p1_x)/ GameConsts.MAX_PLAYER_X
        self.normalized_p2_2_y = (self.p2_2_y - self.p1_y)/ GameConsts.MAX_PLAYER_Y
        self.normalized_g2_x = (self.g2_x - self.p1_x)/ GameConsts.MAX_PLAYER_X
        self.normalized_g2_y = (self.g2_y - self.p1_y)/ GameConsts.MAX_PLAYER_Y
        self.normalized_puck_x = (self.puck_x - self.p1_x)/ GameConsts.MAX_PUCK_X
        self.normalized_puck_y = (self.puck_y - self.p1_y)/ GameConsts.MAX_PUCK_Y
        self.normalized_player_haspuck = 1.0 if self.player_haspuck else 0.0
        self.normalized_goalie_haspuck = 1.0 if self.goalie_haspuck else 0.0


        self.normalized_p1_velx = self.p1_vel_x  / GameConsts.MAX_VEL_XY
        self.normalized_p1_vely = self.p1_vel_y  / GameConsts.MAX_VEL_XY
        self.normalized_p1_2_velx = self.p1_2_vel_x  / GameConsts.MAX_VEL_XY
        self.normalized_p1_2_vely = self.p1_2_vel_y  / GameConsts.MAX_VEL_XY
        self.normalized_p2_velx = (self.p2_vel_x - self.p1_vel_x) / GameConsts.MAX_VEL_XY
        self.normalized_p2_vely = (self.p2_vel_y - self.p1_vel_y)   / GameConsts.MAX_VEL_XY
        self.normalized_p2_2_velx = (self.p2_2_vel_x - self.p1_vel_x)  / GameConsts.MAX_VEL_XY
        self.normalized_p2_2_vely = (self.p2_2_vel_y - self.p1_vel_y) / GameConsts.MAX_VEL_XY
        self.normalized_puck_velx = (self.puck_vel_x - self.p1_vel_x)  / GameConsts.MAX_VEL_XY
        self.normalized_puck_vely = (self.puck_vel_y - self.p1_vel_y) / GameConsts.MAX_VEL_XY
