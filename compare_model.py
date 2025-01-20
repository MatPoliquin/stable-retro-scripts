"""
Compare models
"""

import sys
import argparse

from common import com_print, init_logger
from envs import init_env
from models import init_model, get_num_parameters

import game_wrappers_mgr as games

def parse_cmdline(argv):
    """parse_cmdline(argv)"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--p1_alg', type=str, default='ppo2')
    parser.add_argument('--p2_alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--model1_desc', type=str, default='CNN')
    parser.add_argument('--model2_desc', type=str, default='CNN')
    parser.add_argument('--env', type=str, default='NHL941on1-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='2')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--load_p2_model', type=str, default='')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--deterministic', default=True, action='store_true')

    args = parser.parse_args(argv)

    return args


def main(argv):
    """main(argv)"""

    args = parse_cmdline(argv[1:])

    logger = init_logger(args)

    games.wrappers.init(args)

    com_print('========= Init =============')
    #play_env = init_play_env(args, 2, True)
    p1_env = init_env(None, 1, args.state, 1, args, use_sticky_action=False)
    p2_env = init_env(None, 1, args.state, 1, args, use_sticky_action=False)

    p1_model = init_model(None, args.load_p1_model, args.p1_alg, args, p1_env, logger)
    p2_model = init_model(None, args.load_p2_model, args.p2_alg, args, p2_env, logger)

    model1_params = get_num_parameters(p1_model)
    model2_params = get_num_parameters(p2_model)

    com_print('========= Start Play Loop ==========')

    p1_state = p1_env.reset()
    p2_state = p2_env.reset()

    p1_actions = []
    p2_actions = []

    display = games.wrappers.compare_model(args, args.model1_desc, args.model2_desc, model1_params, model2_params, None)

    while True:
        p1_actions = p1_model.predict(p1_state)
        p2_actions = p2_model.predict(p2_state)

        #play_env.p1_action_probabilities = get_model_probabilities(p1_model, state)[0]
        #play_env.p2_action_probabilities = get_model_probabilities(p2_model, state)[0]

        p1_state, p1_reward, p1_done, p1_info = p1_env.step(p1_actions[0])
        p2_state, p2_reward, p2_done, p2_info = p2_env.step(p2_actions[0])

        display.run_frame(p1_env.render(), p2_env.render())


        if p1_done:
            state = p1_env.reset()

        if p2_done:
            state = p2_env.reset()


if __name__ == '__main__':
    main(sys.argv)
