"""
Game specfic training script for NHL94 1 on 1 on Genesis
"""

#hack to disable flood of warnings
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import datetime
import argparse
import logging
import numpy as np
import retro

from model_trainer import ModelTrainer
from model_vs_game import ModelVsGame

from common import get_model_file_name, com_print, init_logger, create_output_dir

import game_wrappers_mgr as games


def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='MlpPolicy')
    parser.add_argument('--nnsize', type=int, default='256')
    parser.add_argument('--model1_desc', type=str, default='MLP')
    parser.add_argument('--env', type=str, default='NHL941on1-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=16)
    parser.add_argument('--num_timesteps', type=int, default=200_000_000)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--model_1', type=str, default='')
    parser.add_argument('--model_2', type=str, default='')
    parser.add_argument('--alg_verbose', default=True, action='store_false')
    parser.add_argument('--info_verbose', default=True, action='store_false')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--fullscreen', default=False, action='store_true')
    parser.add_argument('--play', default=False, action='store_true')


    args = parser.parse_args(argv)

    return args


# Later plan to add state with free puck for both models
# note that randomization of starting positions is now done in code so no need to add more states (except for free puck which is a special case)
game_states_score_opp = [
    'PenguinsVsSenators.FrontOfNet',
]

game_states_defensezone = [
    'PenguinsVsSenators.dzone'
]


def TrainStates(states, args, logger, rf):
    for state in states:
        com_print('TRAINING ON STATE:%s - %d timesteps' % (state, args.num_timesteps))
        args.state = state
        #args.load_p1_model = p1_model_path
        args.rf = rf
        trainer = ModelTrainer(args, logger)
        p1_model_path = trainer.train()
        return p1_model_path


def main(argv):

    args = parse_cmdline(argv[1:])

    logger = init_logger(args)

    games.wrappers.init(args)

    com_print('================ NHL94 1 on 1 trainer ================')
    com_print('These states will be trained on:')
    com_print(game_states_score_opp)
    com_print(game_states_defensezone)

    # turn off verbose
    args.alg_verbose = False

    args.model_2 =  TrainStates(game_states_score_opp, args, logger, "ScoreGoal")
    args.model_1 = TrainStates(game_states_defensezone, args, logger, "DefenseZone")

    #args.load_p1_model = TrainStates(game_states_losspuck, args, logger, "GetPuck")

    print("----------Trained Models----------")
    print(args.model_1)
    print(args.model_2)
    #print(args.load_p1_model)
    print("----------------------------------")

    if args.play:
        args.state = 'PenguinsVsSenators'
        args.num_timesteps = 1000000
        args.rf = "General"

        player = ModelVsGame(args, logger)

        player.play(continuous=True, need_reset=False)


if __name__ == '__main__':
    main(sys.argv)
