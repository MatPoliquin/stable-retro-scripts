"""
Game specfic training script for WWF Wrestlemania The Arcade Game on Genesis
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

from trainer import ModelTrainer

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--env', type=str, default='WWFArcade-Genesis')
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


game_states = {
    'VeryHard_Yokozuna-01',
    'VeryHard_Yokozuna-02',
    'VeryHard_Yokozuna-03',
    'VeryHard_Yokozuna-04',
    'VeryHard_Yokozuna-05',
    'VeryHard_Yokozuna-06',
    'VeryHard_Yokozuna-07'
}


def main(argv):
    
    args = parse_cmdline(argv[1:])
    
    # Loop through all states

    logger.log('WWF trainer')
    logger.log('These states will be trained on:')
    for state in game_states:
        logger.log(state)

    # turn off verbose
    args.alg_verbose = False

    p1_model_path = args.load_p1_model
    for state in game_states:
        args.state = state
        args.load_p1_model = p1_model_path
        trainer = ModelTrainer(args)
        p1_model_path = trainer.train()
    
    if args.play:
        args.state = 'VeryHard_Yokozuna-01'
        args.load_p1_model = p1_model_path
        args.num_timesteps = 0
        trainer = ModelTrainer(args)
        #trainer.train()
        trainer.play()


if __name__ == '__main__':
    main(sys.argv)