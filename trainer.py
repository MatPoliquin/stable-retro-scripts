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

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--env', type=str, default='SubmarineAttack-Sms')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=24)
    parser.add_argument('--num_timesteps', type=int, default=2000000)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--play', default=False, action='store_true')

    print(argv)
    args = parser.parse_args(argv)

    logger.log("=========== Params ===========")
    logger.log(argv[1:])

    return args

def main(argv):
    
    logger.log('========= Init =============')
    args = parse_cmdline(argv[1:])
    output_fullpath = create_output_dir(args)
    model_savefile_name = get_model_file_name(args)

    env = init_env(output_fullpath, args.num_env, args.state, 1, args)
    p1_model = init_model(None, args.load_p1_model, args.alg, args, env)

    logger.log('========= Start Training ==========')
    p1_model.learn(total_timesteps=args.num_timesteps)
    logger.log('========= End Training ==========')
    p1_model.save(os.path.join(output_fullpath, model_savefile_name))
    logger.log('Mode saved too:%s' % model_savefile_name)
    
    if args.play:
        logger.log('========= Start Play Loop ==========')
        state = env.reset()
        while True:
            env.render()

            p1_actions = p1_model.predict(state)
            
            state, reward, done, info = env.step(p1_actions[0])

            #if done:
            #    state = play_env.reset()


if __name__ == '__main__':
    main(sys.argv)