"""
NHL AI
"""

import math
import random
from game_wrappers.nhl94_const import GameConsts
from game_wrappers.nhl94_gamestate import NHL94GameState
from models import init_model, print_model_info, get_num_parameters, get_model_probabilities


MODEL_NONE = 0 #code
MODEL_DEFENSEZONE = 1
MODEL_SCOREOPP = 2

class NHL94AISystem():
    def __init__(self, args, env, logger):
        self.test = 0
        self.args = args
        self.logger = logger
        self.env = env

        self.target_xy = [0,0]
        self.scoregoal_state = None
        self.pass_button_pressed = False
        self.shooting = False

        self.game_state = NHL94GameState()

        self.models = [None, None, None]
        self.model_params = [None, None, None]
        self.display_probs = (1,0,0,0,0,0,0,0,0,0,0,0,0)
        self.model_in_use = 0
        self.num_models = 0

    def SetModels(self, model_paths):
        i=1
        self.num_models = 0
        for p in model_paths:
            if (p != ''):
                self.models[i] = init_model(None, p, self.args.alg, self.args, self.env, self.logger)
                self.model_params[i] = get_num_parameters(self.models[i])
                self.num_models += 1
                self.model_in_use = i
            i += 1

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

    def Predict(self, model_index, model_input, deterministic):
        p1_actions = self.models[model_index].predict(model_input, deterministic=deterministic)[0][0]
        self.display_probs = get_model_probabilities(self.models[model_index], model_input)[0]
        self.model_in_use = model_index

        return p1_actions

    def Think_TwoModels(self, model_input, state, deterministic):
        p1_actions = [0] * GameConsts.INPUT_MAX

        self.model_in_use = MODEL_NONE

        if state.player_haspuck == True:
            # If in attack zone use ScoreGoal model
            # Otherwise go to attack zone
            if state.p1_y >= GameConsts.ATACKZONE_POS_Y:
                p1_actions = self.Predict(MODEL_SCOREOPP, model_input, deterministic)
                p1_actions[GameConsts.INPUT_C] = 0
                p1_actions[GameConsts.INPUT_B] = 0
                # check if we are in good scoring opportunity
                if state.p1_y < 230 and state.p1_y > 210:
                    if state.p1_vel_x >= 30 and state.puck_x > -23 and state.puck_x < 0:
                        p1_actions[GameConsts.INPUT_C] = 1
                        self.shooting = True
                    elif state.p1_vel_x <= -30 and state.puck_x < 23 and state.puck_x > 0:
                        p1_actions[GameConsts.INPUT_C] = 1
                        self.shooting = True
            else:
                self.GotoTarget(p1_actions, [state.p1_x - 0, state.p1_y - 99])

        elif state.goalie_haspuck:
            p1_actions[GameConsts.INPUT_B] = 1

        else:
            self.shooting = False

            if state.p1_y < GameConsts.DEFENSEZONE_POS_Y and state.p2_haspuck:
                p1_actions = self.Predict(MODEL_DEFENSEZONE, model_input, deterministic)
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
        else:
            self.scoregoal_state = None
            self.GotoTarget(p1_actions, pp_vec)
            self.pass_button_pressed = False

        return p1_actions

    def predict(self, state, info, deterministic):
        if info is None:
            p1_actions = [[0] * GameConsts.INPUT_MAX]
            return p1_actions

        self.game_state.BeginFrame(info[0])

        if self.num_models == 1:
            p1_actions = self.Predict(self.model_in_use, state, deterministic)
            p1_actions[GameConsts.INPUT_MODE] = 0
        elif self.models[1] and self.models[2]:
            p1_actions = [self.Think_TwoModels(state, self.game_state, deterministic)]
        else:
            p1_actions = [self.Think_testAI(self.game_state)]

        self.game_state.EndFrame()

        #self.display_probs = tuple(p1_actions[0])

        return p1_actions
