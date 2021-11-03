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
import pygame
from stable_baselines import logger

from common import init_env, init_model, init_play_env, get_model_file_name, print_model_info, get_num_parameters
from display import GameDisplay

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--model_desc', type=str, default='CNN')
    parser.add_argument('--env', type=str, default='SuperMarioBros3-Nes')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--deterministic', default=True, action='store_true')
    #parser.add_argument('--useframeskip', default=False, action='store_true')

    args = parser.parse_args(argv)

    logger.log("=========== Params ===========")
    logger.log(argv[1:])

    return args

class ModelVsGame:
    def __init__(self, args, need_display=True):

        self.p1_env = init_env(None, 1, args.state, 1, args, True)
        self.display_env, self.uw_display_env = init_play_env(args)
        self.p1_model = init_model(None, args.load_p1_model, args.alg, args, self.p1_env)
        self.need_display = need_display
        self.args = args

        self.uw_display_env.num_params = get_num_parameters(self.p1_model)

    def play(self, continuous=True, need_reset=True):
        state = self.display_env.reset()
        #print(state)
        total_rewards = 0
        skip_frames = 0
        p1_actions = []

        while True:
            p1_actions = self.p1_model.predict(state, deterministic=self.args.deterministic)

            self.uw_display_env.action_probabilities = self.p1_model.action_probability(state)

            #print(self.p1_model.action_probability(state))
            
            state, reward, done, info = self.display_env.step(p1_actions[0])
            total_rewards += reward

            if done:
                if continuous:
                    if need_reset:
                        state = self.display_env.reset()
                else:
                    return info, total_rewards



def main(argv):
    logger.log('========= Init =============')
    args = parse_cmdline(argv[1:])

    player = ModelVsGame(args)

    logger.log('========= Start of Game Loop ==========')
    logger.log('Press ESC or Q to quit')
    player.play(need_reset=False)

if __name__ == '__main__':
    main(sys.argv)