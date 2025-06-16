"""
NHL AI
"""

import math
import random
from game_wrappers.nhl94_const import GameConsts
from game_wrappers.nhl94_gamestate import NHL94GameState
from models_utils import init_model, get_num_parameters, get_model_probabilities


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

        if args.env == 'NHL941on1-Genesis':
            self.game_state = NHL94GameState(1)
        else:
            self.game_state = NHL94GameState(2)

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

        t1 = state.team1
        t2 = state.team2

        if t1.player_haspuck == True:
            # If in attack zone use ScoreGoal model
            # Otherwise go to attack zone
            if t1.players[0].y >= GameConsts.ATACKZONE_POS_Y:
                p1_actions = self.Predict(MODEL_SCOREOPP, model_input, deterministic)
                p1_actions[GameConsts.INPUT_C] = 0
                p1_actions[GameConsts.INPUT_B] = 0
                # check if we are in good scoring opportunity
                if t1.players[0].y < 230 and t1.players[0].y > 210:
                    if t1.players[0].vx >= 30 and t1.players[0].vx > -23 and state.puck.x < 0:
                        p1_actions[GameConsts.INPUT_C] = 1
                        self.shooting = True
                    elif t1.players[0].vx <= -30 and t1.players[0].vx < 23 and state.puck.x > 0:
                        p1_actions[GameConsts.INPUT_C] = 1
                        self.shooting = True
            else:
                self.GotoTarget(p1_actions, [t1.players[0].x - 0, t1.players[0].y - 99])

        elif t1.goalie_haspuck:
            p1_actions[GameConsts.INPUT_B] = 1

        else:
            self.shooting = False

            if t1.players[0].y < GameConsts.DEFENSEZONE_POS_Y and t2.player_haspuck:
                p1_actions = self.Predict(MODEL_DEFENSEZONE, model_input, deterministic)
            else:
                pp_vec = [t1.players[0].x - state.puck.x, t1.players[0].y - state.puck.y]
                self.GotoTarget(p1_actions, pp_vec)

        if self.shooting == True:
            p1_actions[GameConsts.INPUT_MODE] = 1
            p1_actions[GameConsts.INPUT_C] = 1

        return p1_actions

    def SelectRandomTarget(self):
        x = (random.random() - 0.5) * 240
        y = (random.random() - 0.5) * 460

        return [x,y]

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

        self.game_state.EndFrame()

        #self.display_probs = tuple(p1_actions[0])

        return p1_actions
