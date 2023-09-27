"""
NHL AI
Very simple hard coded AI for testing purposes
"""

import math
from game_wrappers.nhl94_const import GameConsts

AI_STATE_IDLE = 0
AI_STATE_ISHOOTING = 1

from models import init_model

class NHL94AISystem():
    def __init__(self, args, env, logger):
        self.test = 0
        self.args = args
        self.use_model = True
        self.p1_model = None
        self.state = AI_STATE_IDLE
        if args.load_p1_model is '':
            self.use_model = False
        else:
            self.p1_model = init_model(None, args.load_p1_model, args.alg, args, env, logger)

        self.get_puck_model = None
        self.score_goal_model = None

        self.logger = logger
        self.env = env

    def SetModels(self, get_puck_model_path, score_goal_model_path):
        if get_puck_model_path != '' and score_goal_model_path != '':
            print(get_puck_model_path)
            print(score_goal_model_path)
            self.get_puck_model = init_model(None, get_puck_model_path, self.args.alg, self.args, self.env, self.logger)
            self.score_goal_model = init_model(None, score_goal_model_path, self.args.alg, self.args, self.env, self.logger)

    def GotoTarget(self, p1_actions, target_vec):
        if target_vec[0] > 0:
            p1_actions[GameConsts.INPUT_LEFT] = 1
        else:
            p1_actions[GameConsts.INPUT_RIGHT] = 1

        if target_vec[1] > 0:
            p1_actions[GameConsts.INPUT_DOWN] = 1
        else:
            p1_actions[GameConsts.INPUT_UP] = 1

    def DistToPos(self, vec1, vec2):
        tmp = (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2
    
        return math.sqrt(tmp)

    def Think_TwoModels(self, info, state, deterministic):

        p1_actions = [0] * GameConsts.INPUT_MAX

        p1_x = info.get('p1_x')
        p1_y = info.get('p1_y')
        g1_x = info.get('g1_x')
        g1_y = info.get('g1_y')
        puck_x = info.get('puck_x')
        puck_y = info.get('puck_y')
        fullstar_x = info.get('fullstar_x')
        fullstar_y = info.get('fullstar_y')


        player_haspuck = False
        goalie_haspuck = False

        if(p1_x == fullstar_x and p1_y == fullstar_y):
            player_haspuck = True
        elif(g1_x == fullstar_x and g1_y == fullstar_y):
            goalie_haspuck = True


        if player_haspuck:
            p1_actions = self.score_goal_model.predict(state, deterministic=deterministic)

        elif goalie_haspuck:
            p1_actions[GameConsts.INPUT_B] = 1
            return [[p1_actions]]
            print('GOALIE PASS')
        else:
            p1_actions = self.get_puck_model.predict(state, deterministic=deterministic)

        return p1_actions


    def Think_testAI(self, info):
        p1_actions = [0] * GameConsts.INPUT_MAX

        p1_x = info.get('p1_x')
        p1_y = info.get('p1_y')
        g1_x = info.get('g1_x')
        g1_y = info.get('g1_y')
        puck_x = info.get('puck_x')
        puck_y = info.get('puck_y')
        fullstar_x = info.get('fullstar_x')
        fullstar_y = info.get('fullstar_y')

        player_haspuck = False
        goalie_haspuck = False

        if(p1_x == fullstar_x and p1_y == fullstar_y):
            player_haspuck = True
        elif(g1_x == fullstar_x and g1_y == fullstar_y):
            goalie_haspuck = True

        pp_vec = [p1_x - puck_x, p1_y - puck_y]
        tmp = (p1_x - puck_x)**2 + (p1_y - puck_y)**2
        pp_dist = math.sqrt(tmp)

        if(goalie_haspuck): print('GOALIE HAS PUCK')

        if player_haspuck:
            dist = self.DistToPos([p1_x, p1_y], [GameConsts.SHOOT_POS_X, GameConsts.SHOOT_POS_Y])

            if dist < 60:
                p1_actions[GameConsts.INPUT_C] = 1
            else:
                self.GotoTarget(p1_actions, [p1_x - GameConsts.SHOOT_POS_X, p1_y - GameConsts.SHOOT_POS_Y])
                print('GOTO SHOOT POSITION')
        elif goalie_haspuck:
            p1_actions[GameConsts.INPUT_B] = 1
            print('GOALIE PASS')
        else:
            self.GotoTarget(p1_actions, pp_vec)
            print('FIND PUCK')

        return [p1_actions]

    def predict(self, state, info, deterministic):
        
        if info is None:
            p1_actions = [[0] * GameConsts.INPUT_MAX]
            return p1_actions
        
        if self.use_model:
            p1_actions = self.p1_model.predict(state, deterministic=deterministic)[0]
        elif self.get_puck_model and self.score_goal_model:
            p1_actions = self.Think_TwoModels(info[0], state, deterministic)[0]
        else:          
            p1_actions = [self.Think_testAI(info[0])[0]]

        return p1_actions

    