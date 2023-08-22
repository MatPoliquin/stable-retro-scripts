"""
AI System
"""

import math
from models import init_model


class AISys():
    def __init__(self, args, env, logger):
        self.test = 0
        self.args = args
        self.use_model = True
        self.p1_model = None
        if args.load_p1_model is '':
            self.use_model = False
        else:
            self.p1_model = init_model(None, args.load_p1_model, args.alg, args, env)


    def predict(self, state, info, deterministic):
        if self.use_model:
            p1_actions = self.p1_model.predict(state, deterministic=deterministic)[0]
        else:
            p1_actions = [0] * GameConsts.INPUT_MAX

        return p1_actions
