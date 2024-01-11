"""
AI System
"""

import math
from models import init_model


class AISys():
    def __init__(self, args, env, logger):
        print('HELLLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
        self.test = 0
        self.args = args
        self.use_model = True
        self.model = None
        
        self.env = env
        self.logger = logger

        

    def SetModels(self, model_paths):
        if model_paths[0] != None:
            self.model = init_model(None, model_paths[0], self.args.alg, self.args, self.env, self.logger)

    def predict(self, state, info, deterministic):
        if self.model:
            p1_actions = self.model.predict(state, deterministic=deterministic)[0]
        else:
            p1_actions = None

        return p1_actions
