import os
import sys
import datetime
import argparse
import logging
import gc
import numpy as np
import retro

from model_trainer import ModelTrainer
from model_vs_game import ModelVsGame

from common import get_model_file_name, com_print, init_logger, create_output_dir

import game_wrappers_mgr as games

NUM_TEST_MATCHS = 10

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--model1_desc', type=str, default='CNN')
    parser.add_argument('--env', type=str, default='MortalKombatII-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=16)
    parser.add_argument('--num_timesteps', type=int, default=10_000_000)
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
    parser.add_argument('--play', default=False, action='store_true')

    args = parser.parse_args(argv)

    return args


game_states= [
    'LiuKangVsBaraka_VeryHard_01',
    'LiuKangVsReptile_VeryHard_02',
    'LiuKangVsJax_VeryHard_03',
    #'LiuKangVsRayden_VeryHard_04',
    #'LiuKangVsKitana_VeryHard_05',
    #'LiuKangVsLiuKang_VeryHard_06',
    #'LiuKangVsSubZero_VeryHard_07',
    #'LiuKangVsMileena_VeryHard_08',
    #'LiuKangVsCage_VeryHard_09',
    #'LiuKangVsKungLao_VeryHard_10',
    #'LiuKangVsScorpion_VeryHard_11',
    #'LiuKangVsJade_VeryHard_12',
    #'LiuKangVsShangTsung_VeryHard_13'
]

def test_model(args, num_matchs, logger):
    game = ModelVsGame(args, logger, need_display=False)

    won_matchs = 0
    total_rewards = 0
    for i in range(0, num_matchs):
        info, reward = game.play(False)
        if info[0].get('enemy_health') == 0:
            won_matchs += 1
        total_rewards += reward
        #print(total_rewards)
        #print(info)


    return won_matchs, total_rewards

def main(argv):
    args = parse_cmdline(argv[1:])

    logger = init_logger(args)

    games.wrappers.init(args)

    com_print('================ MK2 trainer ================')
    com_print('These states will be trained on:')
    com_print(game_states)

    # turn off verbose
    args.alg_verbose = False

    p1_model_path = args.load_p1_model

    # Train model on each state
    if not args.test_only:
        for state in game_states:
            com_print('TRAINING ON STATE:%s - %d timesteps' % (state, args.num_timesteps))
            args.state = state
            args.load_p1_model = p1_model_path
            trainer = ModelTrainer(args, logger)
            p1_model_path = trainer.train()

            gc.collect()

            # Test model performance
            #num_test_matchs = NUM_TEST_MATCHS
            #new_args = args
            #new_args.model_1 = p1_model_path
            #new_args.model_2 = ''
            #com_print('    TESTING MODEL ON %d matchs...' % num_test_matchs)
            #won_matchs, total_reward = test_model(new_args, num_test_matchs, logger)
            #percentage = won_matchs / num_test_matchs
            #com_print('    WON MATCHS:%d/%d - ratio:%f' % (won_matchs, num_test_matchs, percentage))
            #com_print('    TOTAL REWARDS:%d\n' %  total_reward)

    # Test performance of model on each state
    com_print('====== TESTING MODEL ======')
    for state in game_states:
        num_test_matchs = NUM_TEST_MATCHS
        new_args = args
        new_args.model_1 = p1_model_path
        new_args.model_2 = None
        won_matchs, total_reward = test_model(new_args, num_test_matchs, logger)
        percentage = won_matchs / num_test_matchs
        com_print('STATE:%s... WON MATCHS:%d/%d TOTAL REWARDS:%d' % (state, won_matchs, num_test_matchs, total_reward))
        gc.collect()

    if args.play:
        args.state = 'LiuKangVsRayden_VeryHard_01'
        args.model_1 = p1_model_path
        args.model_2 = ''
        args.num_timesteps = 0

        player = ModelVsGame(args, logger, True)

        player.play(continuous=True, need_reset=False)


if __name__ == '__main__':
    main(sys.argv)
