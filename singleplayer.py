"""
Play a pre-trained model on a retro env
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

from common import init_env, init_model, init_play_env, get_model_file_name, GameDisplay

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--p1_alg', type=str, default='ppo2')
    parser.add_argument('--env', type=str, default='MortalKombatII-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--noserver', default=False, action='store_true')

    print(argv)
    args = parser.parse_args(argv)

    logger.log("=========== Params ===========")
    logger.log(argv[1:])

    return args

def main(argv):
    
    logger.log('========= Init =============')
    args = parse_cmdline(argv[1:])

    play_env = init_play_env(args)
    p1_env = init_env(None, 1, None, 1, args)
    p1_model = init_model(None, args.load_p1_model, args.p1_alg, args, p1_env)

    logger.log('========= Start Game Loop ==========')
    state = play_env.reset()
    while True:
        play_env.render()

        p1_actions = p1_model.predict(state)
            
        state, reward, done, info = play_env.step(p1_actions[0])

        if done:
            state = play_env.reset()

if __name__ == '__main__':
    main(sys.argv)