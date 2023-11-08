"""
Train a Model on NHL 94
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import retro
import datetime
import argparse
import logging
import numpy as np

from common import get_model_file_name, com_print, init_logger, create_output_dir
from models import init_model, print_model_info, get_num_parameters
from envs import init_env, init_play_env


def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--nnsize', type=int, default='256')
    parser.add_argument('--env', type=str, default='NHL941on1-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=24)
    parser.add_argument('--num_timesteps', type=int, default=6000000)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--alg_verbose', default=True, action='store_true')
    parser.add_argument('--info_verbose', default=True, action='store_true')
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--rf', type=str, default='')

    print(argv)
    args = parser.parse_args(argv)

    #if args.info_verbose is False:
    #    logger.set_level(logger.DISABLED)

    return args

class ModelTrainer:
    def __init__(self, args, logger):
        self.args = args

        #if self.args.alg_verbose:
        #    logger.log('========= Init =============')
        
        self.output_fullpath = create_output_dir(args)
        model_savefile_name = get_model_file_name(args)
        self.model_savepath = os.path.join(self.output_fullpath, model_savefile_name)

        self.env = init_env(self.output_fullpath, args.num_env, args.state, args.num_players, args)
        
        self.p1_model = init_model(self.output_fullpath, args.load_p1_model, args.alg, args, self.env, logger)
      
        #if self.args.alg_verbose:
        com_print('OUTPUT PATH:   %s' % self.output_fullpath)
        com_print('ENV:           %s' % args.env)
        com_print('STATE:         %s' % args.state)
        com_print('NN:            %s' % args.nn)
        com_print('ALGO:          %s' % args.alg)
        com_print('NUM TIMESTEPS: %s' % args.num_timesteps)
        com_print('NUM ENV:       %s' % args.num_env)
        com_print('NUM PLAYERS:   %s' % args.num_players)

        print(self.env.observation_space)

    def train(self):
        #if self.args.alg_verbose:
        com_print('========= Start Training ==========')
        self.p1_model.learn(total_timesteps=self.args.num_timesteps)
        #if self.args.alg_verbose:
        com_print('========= End Training ==========')

        self.p1_model.save(self.model_savepath )
        #if self.args.alg_verbose:
        com_print('Model saved to:%s' % self.model_savepath)

        return self.model_savepath

    def play(self, continuous=True):
        #if self.args.alg_verbose:
        com_print('========= Start Play Loop ==========')
        state = self.env.reset()
        while True:
            self.env.render(mode='human')

            p1_actions = self.p1_model.predict(state)
            
            state, reward, done, info = self.env.step(p1_actions[0])

            #print(reward)

            if done[0]:
                state = self.env.reset()

            if not continuous and done is True:
                return info



def main(argv):
    
    args = parse_cmdline(argv[1:])

    logger = init_logger(args)
    com_print("=========== Params ===========")
    com_print(args)
    
    trainer = ModelTrainer(args, logger)

    trainer.train()
    
    if args.play:
        trainer.play()


if __name__ == '__main__':
    main(sys.argv)