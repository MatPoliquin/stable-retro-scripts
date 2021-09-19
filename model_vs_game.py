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

from common import init_env, init_model, init_play_env, get_model_file_name
from display import GameDisplay

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--env', type=str, default='WWFArcade-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--noserver', default=False, action='store_true')

    args = parser.parse_args(argv)

    logger.log("=========== Params ===========")
    logger.log(argv[1:])

    return args

class ModelVsGame:
    def __init__(self, args, need_display=True):
        self.play_env = init_play_env(args)
        self.p1_env = init_env(None, 1, args.state, 1, args)
        self.p1_model = init_model(None, args.load_p1_model, args.alg, args, self.p1_env)
        self.need_display = need_display

        if need_display:
            self.display = GameDisplay(args) 

    def play(self, continuous=True, need_reset=True):
        #logger.log('========= Start of Game Loop ==========')
        #logger.log('Press ESC or Q to quit')
        state = self.play_env.reset()
        total_rewards = 0
        while True:
            if self.need_display:
                framebuffer = self.play_env.render(mode='rgb_array')
                self.display.draw_frame(framebuffer)

            #p1_actions = self.p1_model.predict(state, deterministic=True)
            p1_actions = self.p1_model.predict(state, deterministic=False)

            #print(p1_actions[0])
            #print(p1_model.action_probability(state))
            
            state, reward, done, info = self.play_env.step(p1_actions[0])
            total_rewards += reward
            #print(info)

            if done:
                if continuous:
                    if need_reset:
                        state = self.play_env.reset()
                else:
                    return info, total_rewards

            if self.need_display:
                keystate = self.display.get_input()
                if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
                    logger.log('Exiting...')
                    break


def main(argv):

    logger.log('========= Init =============')
    args = parse_cmdline(argv[1:])

    player = ModelVsGame(args)

    player.play()

if __name__ == '__main__':
    main(sys.argv)