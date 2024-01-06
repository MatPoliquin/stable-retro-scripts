"""
Play a pre-trained model on NHL 94
"""

import os
import sys
import retro
import datetime
import argparse
import logging
import numpy as np
import pygame
from common import get_model_file_name, com_print, init_logger
from models import print_model_info, get_num_parameters, get_model_probabilities
from envs import init_env, init_play_env

import game_wrappers_mgr as games

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--model1_desc', type=str, default='CNN')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--nnsize', type=int, default='256')
    parser.add_argument('--env', type=str, default='NHL941on1-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--model_1', type=str, default='')
    parser.add_argument('--model_2', type=str, default='')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--fullscreen', default=False, action='store_true')
    parser.add_argument('--rf', type=str, default='')
    #parser.add_argument('--useframeskip', default=False, action='store_true')

    args = parser.parse_args(argv)

    return args

class ModelVsGame:
    def __init__(self, args, logger, need_display=True):

        self.p1_env = init_env(None, 1, args.state, 1, args, True)
        self.display_env = init_play_env(args, 1, False, need_display, False)

        self.ai_sys = games.wrappers.ai_sys(args, self.p1_env, logger)
        if args.model_1 != '':
            self.ai_sys.SetModels(args.model_1, args.model_2)
        
        self.need_display = need_display
        self.args = args

        if self.ai_sys.p1_model is not None:
            self.display_env.num_params = get_num_parameters(self.ai_sys.p1_model)

    def play(self, continuous=True, need_reset=True):
        state = self.display_env.reset()

        total_rewards = 0
        skip_frames = 0
        p1_actions = []
        info = None

        while True:
            p1_actions = self.ai_sys.predict(state, info=info, deterministic=self.args.deterministic)
            if self.ai_sys.p1_model is not None:
                self.display_env.action_probabilities = get_model_probabilities(self.ai_sys.p1_model, state)[0]
            else:
                self.display_env.action_probabilities = []
            
            for i in range(4):
                state, reward, done, info = self.display_env.step(p1_actions)
                total_rewards += reward

            if done:
                if continuous:
                    if need_reset:
                        state = self.display_env.reset()
                else:
                    return info, total_rewards



def main(argv):
    args = parse_cmdline(argv[1:])

    logger = init_logger(args)

    games.wrappers.init(args)

    player = ModelVsGame(args, logger)

    com_print('========= Start of Game Loop ==========')
    com_print('Press ESC or Q to quit')
    player.play(need_reset=False)

if __name__ == '__main__':
    main(sys.argv)