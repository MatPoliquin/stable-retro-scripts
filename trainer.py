"""
Train a Model on a retro env
"""

import os
import sys
import retro
import datetime
import joblib
import argparse
import logging
import numpy as np
from stable_baselines import logger

from common import init_env, init_model, init_play_env, get_model_file_name, create_output_dir

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--env', type=str, default='SubmarineAttack-Sms')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=24)
    parser.add_argument('--num_timesteps', type=int, default=100000)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--alg_verbose', default=True, action='store_true')
    parser.add_argument('--play', default=False, action='store_true')

    print(argv)
    args = parser.parse_args(argv)

    logger.log("=========== Params ===========")
    logger.log(argv[1:])

    return args

class ModelTrainer:
    def __init__(self, args):
        logger.log('========= Init =============')
        self.args = args

        self.output_fullpath = create_output_dir(args)
        model_savefile_name = get_model_file_name(args)
        self.model_savepath = os.path.join(self.output_fullpath, model_savefile_name)

        self.env = init_env(self.output_fullpath, args.num_env, args.state, 1, args)
        self.p1_model = init_model(None, args.load_p1_model, args.alg, args, self.env)

        logger.log('OUTPUT PATH:   %s' % self.output_fullpath)
        logger.log('ENV:           %s' % args.env)
        logger.log('STATE:         %s' % args.state)
        logger.log('NN:            %s' % args.nn)
        logger.log('ALGO:          %s' % args.alg)
        logger.log('NUM TIMESTEPS: %s' % args.num_timesteps)
        logger.log('NUM ENV:       %s' % args.num_env)
        logger.log('NUM PLAYERS:   %s' % args.num_players)

    def train(self):
        logger.log('========= Start Training ==========')
        self.p1_model.learn(total_timesteps=self.args.num_timesteps)
        logger.log('========= End Training ==========')

        self.p1_model.save(self.model_savepath )
        logger.log('Mode saved too:%s' % self.model_savepath)

        return self.model_savepath

    def play(self):
        logger.log('========= Start Play Loop ==========')
        state = self.env.reset()
        while True:
            self.env.render()

            p1_actions = self.p1_model.predict(state)
            
            state, reward, done, info = self.env.step(p1_actions[0])



def main(argv):
    
    args = parse_cmdline(argv[1:])
    
    trainer = ModelTrainer(args)

    trainer.train()
    
    if args.play:
        trainer.play()


if __name__ == '__main__':
    main(sys.argv)