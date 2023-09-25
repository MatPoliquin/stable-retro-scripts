"""
Game specfic training script for NHL94 1 on 1 on Genesis
"""

#hack to disable flood of warnings
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import retro
import datetime
import joblib
import argparse
import logging
import numpy as np

from model_trainer import ModelTrainer
from model_vs_game import ModelVsGame

from common import get_model_file_name, com_print, init_logger, create_output_dir

import game_wrappers_mgr as games

NUM_TEST_MATCHS = 10

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
    parser.add_argument('--num_timesteps', type=int, default=15_000_000)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--alg_verbose', default=True, action='store_false')
    parser.add_argument('--info_verbose', default=True, action='store_false')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--play', default=False, action='store_true')

    args = parser.parse_args(argv)

    return args


game_states_gotpuck = [
    'PenguinsVsSenators.AttackZone01',
    'PenguinsVsSenators.AttackZone02',
    'PenguinsVsSenators.AttackZone03',
    'PenguinsVsSenators.AttackZone04',
    'PenguinsVsSenators.AttackZone05',
    'PenguinsVsSenators.AttackZone06',
    'PenguinsVsSenators.AttackZone07',
    'PenguinsVsSenators.gotpuck01',
    'PenguinsVsSenators.gotpuck02',
    'PenguinsVsSenators.gotpuck03',
    'PenguinsVsSenators.gotpuck04',
    'PenguinsVsSenators.gotpuck05',
    'PenguinsVsSenators.gotpuck06',
    'PenguinsVsSenators.gotpuck07'
]

game_states_losspuck = [
    'PenguinsVsSenators.losspuck01',
    'PenguinsVsSenators.losspuck02',
    'PenguinsVsSenators.losspuck03',
    'PenguinsVsSenators.losspuck04',
    'PenguinsVsSenators.losspuck05',
    'PenguinsVsSenators.losspuck06',
    'PenguinsVsSenators.losspuck07'
]






def main(argv):
    
    args = parse_cmdline(argv[1:])

    logger = init_logger(args)

    games.wrappers.init(args)
    
    com_print('================ NHL94 1 on 1 trainer ================')
    com_print('These states will be trained on:')
    com_print(game_states_losspuck)

    # turn off verbose
    args.alg_verbose = False
    
    p1_model_path = args.load_p1_model

    # Train model on each state
    if not args.test_only:    
        for state in game_states_losspuck:
            com_print('TRAINING ON STATE:%s - %d timesteps' % (state, args.num_timesteps))
            args.state = state
            args.load_p1_model = p1_model_path
            trainer = ModelTrainer(args, logger)
            p1_model_path = trainer.train()

            # Test model performance
            #num_test_matchs = NUM_TEST_MATCHS
            #new_args = args
            #new_args.load_p1_model = p1_model_path
            #com_print('    TESTING MODEL ON %d matchs...' % num_test_matchs)
            #won_matchs, total_reward = test_model(new_args, num_test_matchs, logger)
            #percentage = won_matchs / num_test_matchs
            #com_print('    WON MATCHS:%d/%d - ratio:%f' % (won_matchs, num_test_matchs, percentage))
            #com_print('    TOTAL REWARDS:%d\n' %  total_reward)

    
    # Test performance of model on each state
    #com_print('====== TESTING MODEL ======')
    #for state in game_states_attackzone:
    #    num_test_matchs = NUM_TEST_MATCHS
    #    new_args = args
    #    won_matchs, total_reward = test_model(new_args, num_test_matchs, logger)
    #    percentage = won_matchs / num_test_matchs
    #    com_print('STATE:%s... WON MATCHS:%d/%d TOTAL REWARDS:%d' % (state, won_matchs, num_test_matchs, total_reward))

    if args.play:
        args.state = 'PenguinsVsSenators.losspuck07'
        args.load_p1_model = p1_model_path
        args.num_timesteps = 0

        player = ModelVsGame(args, logger)

        player.play(continuous=True, need_reset=False)


if __name__ == '__main__':
    main(sys.argv)
