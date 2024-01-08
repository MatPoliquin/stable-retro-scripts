"""
NHL AI
Very simple hard coded AI for testing purposes
"""

import math
import random
from game_wrappers.nhl94_const import GameConsts
from game_wrappers.nhl94_gamestate import NHL94GameState
from models import init_model, print_model_info, get_num_parameters, get_model_probabilities

AI_STATE_IDLE = 0
AI_STATE_ISHOOTING = 1


class NHL94AISystem():
    def __init__(self, args, env, logger):
        self.test = 0
        self.args = args
        self.use_model = False
        self.p1_model = None
        self.state = AI_STATE_IDLE
        self.logger = logger
        self.env = env

        self.get_puck_model = None
        self.score_goal_model = None

        self.target_xy = [0,0]

        self.scoregoal_state = None
        self.pass_button_pressed = False

        self.shooting = False

        # for display
        self.display_probs = (1,0,0,0,0,0,0,0,0,0,0,0,0)
        self.model_1_num_params = 0
        self.model_2_num_params = 0
        self.model_in_use = 0

        self.game_state = NHL94GameState()

        

    def SetModels(self, get_puck_model_path, score_goal_model_path):
        if get_puck_model_path != '' and score_goal_model_path != '':
            print(get_puck_model_path)
            print(score_goal_model_path)
            self.get_puck_model = init_model(None, get_puck_model_path, self.args.alg, self.args, self.env, self.logger)
            self.score_goal_model = init_model(None, score_goal_model_path, self.args.alg, self.args, self.env, self.logger)

            self.model_1_num_params = get_num_parameters(self.get_puck_model)
            self.model_2_num_params = get_num_parameters(self.score_goal_model)
    
    def SetModel(self, model_path):
        if model_path != '':
            self.use_model = True
            self.p1_model = init_model(None, model_path, self.args.alg, self.args, self.env, self.logger)
            

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

    def Think_TwoModels(self, model_input, state, deterministic):

        p1_actions = [0] * GameConsts.INPUT_MAX

        self.model_in_use = 0

        if state.player_haspuck == True:
            # If in attack zone use ScoreGoal model
            # Otherwise go to attack zone
            if state.p1_y >= 100:
                p1_actions = self.score_goal_model.predict(model_input, deterministic=deterministic)[0][0]
                self.model_in_use = 2
                #print(tuple(p1_actions))
                self.display_probs = get_model_probabilities(self.score_goal_model, model_input)[0]
                #print(self.display_probs)
                p1_actions[GameConsts.INPUT_C] = 0
                p1_actions[GameConsts.INPUT_B] = 0
                # check if we are in good scoring opportunity
                if state.p1_y < 230 and state.p1_y > 210:
                    if state.p1_vel_x >= 30 and state.puck_x > -23 and state.puck_x < 0:
                        p1_actions[GameConsts.INPUT_C] = 1
                        self.shooting = True
                        #print('SHOOT')
                    elif state.p1_vel_x <= -30 and state.puck_x < 23 and state.puck_x > 0:
                        p1_actions[GameConsts.INPUT_C] = 1
                        self.shooting = True
                        #print('SHOOT')
                #else:
                #    p1_actions = self.score_goal_model.predict(model_input, deterministic=deterministic)[0][0]
                    #print(p1_actions)
            else:
                #dist = self.DistToPos([state.p1_x, state.p1_y], [GameConsts.SHOOT_POS_X, GameConsts.SHOOT_POS_Y])
                self.GotoTarget(p1_actions, [state.p1_x - 0, state.p1_y - 99])

        elif state.goalie_haspuck:
            p1_actions[GameConsts.INPUT_B] = 1

        else:
            self.shooting = False

            if state.p1_y < -80 and state.p2_haspuck:
                p1_actions = self.get_puck_model.predict(model_input, deterministic=deterministic)[0][0]
                self.model_in_use = 1
                self.display_probs = get_model_probabilities(self.get_puck_model, model_input)[0]
            else:
                pp_vec = [state.p1_x - state.puck_x, state.p1_y - state.puck_y]
                self.GotoTarget(p1_actions, pp_vec)

        if self.shooting == True:
            p1_actions[GameConsts.INPUT_MODE] = 1
            p1_actions[GameConsts.INPUT_C] = 1

        return p1_actions

    def SelectRandomTarget(self):
        x = (random.random() - 0.5) * 240
        y = (random.random() - 0.5) * 460

        return [x,y]

    def Think_GotoRandomTarget(self, state):
        p1_actions = [0] * GameConsts.INPUT_MAX

        pp_vec = [state.p1_x - state.puck_x, state.p1_y - state.puck_y]
        tmp = (state.p1_x - state.puck_x)**2 + (state.p1_y - state.puck_y)**2
        pp_dist = math.sqrt(tmp)

        #if(state.goalie_haspuck): print('GOALIE HAS PUCK')

        if state.player_haspuck:
            dist = self.DistToPos([state.p1_x, state.p1_y], self.target_xy)

            if dist < 20.0:
                self.target_xy = self.SelectRandomTarget()
                #print(self.target_xy)
                r = random.random()
                #print(r)
                if r < 0.2:
                    p1_actions[GameConsts.INPUT_B] = 1

            # TODO check if target is near opposing player
            self.GotoTarget(p1_actions, [state.p1_x - self.target_xy[0], state.p1_y - self.target_xy[1]])

        elif state.goalie_haspuck:
            p1_actions[GameConsts.INPUT_B] = 1
            #print('GOALIE PASS')
        else:
            self.GotoTarget(p1_actions, pp_vec)
            #print('FIND PUCK')

        return p1_actions
    
    def Think_ScoreGoal01(self, state):
        p1_actions = [0] * GameConsts.INPUT_MAX

        dist = self.DistToPos([state.p1_x, state.p1_y], [GameConsts.SHOOT_POS_X, GameConsts.SHOOT_POS_Y])

        if dist < 60:
            p1_actions[GameConsts.INPUT_C] = 1
        else:
            self.GotoTarget(p1_actions, [state.p1_x - GameConsts.SHOOT_POS_X, state.p1_y - GameConsts.SHOOT_POS_Y])

        return p1_actions

        
    def Think_ScoreGoal02(self, state):
        p1_actions = [0] * GameConsts.INPUT_MAX

        if self.scoregoal_state == None:
            #choose target on left side
            dist = self.DistToPos([state.p1_x, state.p1_y], [90, 180])

            if dist < 50:
                self.scoregoal_state = "On left side"
            else:
                self.GotoTarget(p1_actions, [state.p1_x - (90), state.p1_y - 180])
                    #print('GOTO SHOOT POSITION')
        
        if self.scoregoal_state == "On left side":
            dist = self.DistToPos([state.p1_x, state.p1_y], [-80, 200])

            if dist < 50:
                self.scoregoal_state = "On right side"
            else:
                self.GotoTarget(p1_actions, [state.p1_x - (-80), state.p1_y - 200])

        if self.scoregoal_state == "On right side":
            p1_actions[GameConsts.INPUT_C] = 1
            self.scoregoal_state = None

        return p1_actions



    def Think_testAI(self, state):
        p1_actions = [0] * GameConsts.INPUT_MAX

        pp_vec = [state.p1_x - state.puck_x, state.p1_y - state.puck_y]
        tmp = (state.p1_x - state.puck_x)**2 + (state.p1_y - state.puck_y)**2
        pp_dist = math.sqrt(tmp)

        #if(state.goalie_haspuck): print('GOALIE HAS PUCK')

        if state.player_haspuck:
            

            p1_actions = self.Think_ScoreGoal02(state)
            self.pass_button_pressed = False
           
        elif state.goalie_haspuck:
            self.scoregoal_state = None
            if not self.pass_button_pressed:
                p1_actions[GameConsts.INPUT_B] = 1
                self.pass_button_pressed = True
            else:
                self.pass_button_pressed = False
            
            #print('GOALIE PASS')
        else:
            self.scoregoal_state = None
            self.GotoTarget(p1_actions, pp_vec)
            self.pass_button_pressed = False
            #print('FIND PUCK')

        #print(p1_actions)

        return p1_actions

    def predict(self, state, info, deterministic):
        if info is None:
            p1_actions = [[0] * GameConsts.INPUT_MAX]
            return p1_actions
        
        self.game_state.BeginFrame(info[0])
        
        if self.use_model:
            p1_actions = self.p1_model.predict(state, deterministic=deterministic)[0]
        elif self.get_puck_model and self.score_goal_model:
            p1_actions = [self.Think_TwoModels(state, self.game_state, deterministic)]
        else:          
            p1_actions = [self.Think_testAI(self.game_state)]

        self.game_state.EndFrame()

        #self.display_probs = tuple(p1_actions[0])

        return p1_actions

    