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

        model_path = ''
        try:
            model_path = args.load_p1_model if args.load_p1_model != '' else args.model_1
        except AttributeError:
            try:
                model_path = args.model_1
            except AttributeError:
                self.use_model = False
                print('No model attribute found')

        if model_path and self.use_model:
            self.p1_model = init_model(None, model_path, args.alg, args, env, logger)

    def predict(self, state, info, deterministic):
        if self.use_model:
            p1_actions = self.p1_model.predict(state, deterministic=deterministic)[0]
        else:
            p1_actions = [0] * GameConsts.INPUT_MAX

        return p1_actions
