"""
Game specfic training script for WWF Wrestlemania The Arcade Game on Genesis
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
from stable_baselines import logger

sys.path.append('.')

from model_trainer import ModelTrainer
from model_vs_game import ModelVsGame


NUM_TEST_MATCHS = 10

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--model_desc', type=str, default='CNN')
    parser.add_argument('--env', type=str, default='WWFArcade-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=24)
    parser.add_argument('--num_timesteps', type=int, default=3000000)
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


    if args.info_verbose is False:
        logger.set_level(logger.DISABLED)

    logger.log("=========== Params ===========")
    logger.log(argv[1:])

    return args


game_states = [
    'VeryEasy_Yokozuna-01',
    'VeryEasy_Yokozuna-02',
    'VeryEasy_Yokozuna-03',
    'VeryEasy_Yokozuna-04',
    'VeryEasy_Yokozuna-05',
    'VeryEasy_Yokozuna-06',
    'VeryEasy_Yokozuna-07'
]

game_states_veryhard = [
    'VeryHard_Yokozuna-01',
    'VeryHard_Yokozuna-02',
    'VeryHard_Yokozuna-03',
    'VeryHard_Yokozuna-04',
    'VeryHard_Yokozuna-05',
    'VeryHard_Yokozuna-06',
    'VeryHard_Yokozuna-07'
]


#game_states = [
#    'VeryEasy_Yokozuna-01'
#]

def test_model(args, num_matchs):
    game = ModelVsGame(args, need_display=False)

    won_matchs = 0
    total_rewards = 0
    for i in range(0, num_matchs):
        info, reward = game.play(False)
        if info.get('won_rounds') == 2:
            won_matchs += 1
        total_rewards += reward
        
        #print(total_rewards)
        #print(info)


    return won_matchs, total_rewards

def main(argv):
    
    args = parse_cmdline(argv[1:])
    
    logger.log('================ WWF trainer ================')
    logger.log('These states will be trained on:')
    logger.log(game_states)

    # turn off verbose
    args.alg_verbose = False
    
    p1_model_path = args.load_p1_model

    # Train model on each state
    if not args.test_only:    
        for state in game_states:
            logger.log('TRAINING ON STATE:%s - %d timesteps' % (state, args.num_timesteps))
            args.state = state
            args.load_p1_model = p1_model_path
            trainer = ModelTrainer(args)
            p1_model_path = trainer.train()

            # Test model performance
            num_test_matchs = NUM_TEST_MATCHS
            new_args = args
            new_args.load_p1_model = p1_model_path
            logger.log('    TESTING MODEL ON %d matchs...' % num_test_matchs)
            won_matchs, total_reward = test_model(new_args, num_test_matchs)
            percentage = won_matchs / num_test_matchs
            logger.log('    WON MATCHS:%d/%d - ratio:%f' % (won_matchs, num_test_matchs, percentage))
            logger.log('    TOTAL REWARDS:%d\n' %  total_reward)

    
    # Test performance of model on each state
    logger.log('====== TESTING MODEL ======')
    for state in game_states:
        num_test_matchs = NUM_TEST_MATCHS
        new_args = args
        won_matchs, total_reward = test_model(new_args, num_test_matchs)
        percentage = won_matchs / num_test_matchs
        logger.log('STATE:%s... WON MATCHS:%d/%d TOTAL REWARDS:%d' % (state, won_matchs, num_test_matchs, total_reward))

    if args.play:
        args.state = 'VeryEasy_Yokozuna-01'
        args.load_p1_model = p1_model_path
        args.num_timesteps = 0

        player = ModelVsGame(args)

        player.play(continuous=True, need_reset=False)


if __name__ == '__main__':
    main(sys.argv)