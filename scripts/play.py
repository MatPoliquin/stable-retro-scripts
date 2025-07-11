"""
Play modes:
- Player vs Model
- Model vs Model
- Model vs Game
"""

import sys
import argparse
import numpy as np
from common import com_print, init_logger
from env_utils import init_env, init_play_env
from models_utils import init_model, get_model_probabilities, get_num_parameters
import game_wrappers_mgr as games

def parse_cmdline(argv):
    parser = argparse.ArgumentParser(description='Play with your model in different modes')

    # Mode selection
    parser.add_argument('--mode', type=str, default='player_vs_model',
                       choices=['player_vs_model', 'model_vs_model', 'model_vs_game'],
                       help='Game mode: player_vs_model, model_vs_model, or model_vs_game')

    # Common arguments
    parser.add_argument('--env', type=str, default='NHL941on1-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default=2)
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--display_width', type=int, default=1920)
    parser.add_argument('--display_height', type=int, default=1080)
    parser.add_argument('--fullscreen', default=False, action='store_true')
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--rf', type=str, default='')
    parser.add_argument('--hyperparams', type=str, default='../hyperparams/default.json')
    parser.add_argument('--video', default=False, action='store_true')
    parser.add_argument('--video_path', type=str, default='../retro_game.avi')

    # Model-related arguments
    parser.add_argument('--alg', type=str, default='ppo2', help='Algorithm for single model')
    parser.add_argument('--p1_alg', type=str, default='ppo2', help='Algorithm for player 1 model')
    parser.add_argument('--p2_alg', type=str, default='ppo2', help='Algorithm for player 2 model')
    parser.add_argument('--nn', type=str, default='MlpPolicy')
    parser.add_argument('--model1_desc', type=str, default='CNN')
    parser.add_argument('--model2_desc', type=str, default='MLP')
    parser.add_argument('--nnsize', type=int, default=256)
    parser.add_argument('--model_1', type=str, default='', help='Model path for player 1')
    parser.add_argument('--model_2', type=str, default='', help='Model path for player 2')
    parser.add_argument('--load_p1_model', type=str, default='', help='Model path for player 1 in model_vs_model mode')
    parser.add_argument('--load_p2_model', type=str, default='', help='Model path for player 2 in model_vs_model mode')

    parser.add_argument('--action_type', type=str, default='FILTERED',
                       choices=['FILTERED', 'DISCRETE', 'MULTI_DISCRETE'],
                       help='Action type: FILTERED, DISCRETE, or MULTI_DISCRETE')

    args = parser.parse_args(argv)

    # Set default num_players based on mode
    if args.mode == 'model_vs_game':
        args.num_players = 1

    return args

class NHL94Player:
    def __init__(self, args, logger, need_display=True):
        self.args = args
        self.logger = logger
        self.need_display = need_display

        if args.mode == 'model_vs_model':
            self.init_model_vs_model()
        else:
            self.init_player_or_game_mode()

    def init_model_vs_model(self):
        """Initialize for model vs model mode"""
        self.play_env = init_play_env(self.args, 2, True)
        self.p1_env = init_env(None, 1, None, 1, self.args, use_sticky_action=False)
        self.p2_env = init_env(None, 1, None, 1, self.args, use_sticky_action=False)

        self.p1_model = init_model(None, self.args.load_p1_model, self.args.p1_alg, self.args, self.p1_env, self.logger)
        self.p2_model = init_model(None, self.args.load_p2_model, self.args.p2_alg, self.args, self.p2_env, self.logger)

        self.play_env.model1_params = get_num_parameters(self.p1_model)
        self.play_env.model2_params = get_num_parameters(self.p2_model)

    def init_player_or_game_mode(self):
        """Initialize for player vs model or model vs game modes"""
        num_players = 1 if self.args.mode == 'model_vs_game' else 2
        self.p1_env = init_env(None, 1, self.args.state, 1, self.args, True)
        self.display_env = init_play_env(self.args, num_players, False, self.need_display, False)

        self.ai_sys = games.wrappers.ai_sys(self.args, self.p1_env, self.logger)
        if self.args.model_1 != '' or self.args.model_2 != '':
            models = [self.args.model_1, self.args.model_2]
            self.ai_sys.SetModels(models)

    def play(self, continuous=True, need_reset=True):
        """Main game loop"""
        if self.args.mode == 'model_vs_model':
            self.play_model_vs_model(continuous, need_reset)
        else:
            self.play_player_or_game_mode(continuous, need_reset)

    def play_model_vs_model(self, continuous, need_reset):
        """Game loop for model vs model mode"""
        state = self.play_env.reset()

        while True:
            p1_actions = self.p1_model.predict(state)[0]  # Get the action array directly
            p2_actions = self.p2_model.predict(state)[0]  # Get the action array directly

            self.play_env.p1_action_probabilities = get_model_probabilities(self.p1_model, state)[0]
            self.play_env.p2_action_probabilities = get_model_probabilities(self.p2_model, state)[0]

            # Combine actions properly for 2-player game
            actions = [p1_actions, p2_actions]

            state, _, done, _ = self.play_env.step(actions)

            if done:
                if continuous:
                    if need_reset:
                        state = self.play_env.reset()
                else:
                    return

    def play_player_or_game_mode(self, continuous, need_reset):
        """Game loop for player vs model or model vs game modes"""
        state = self.display_env.reset()
        total_rewards = 0
        info = None

        while True:
            # Get actions from AI system
            p1_actions = self.ai_sys.predict(state, info=info, deterministic=self.args.deterministic)

            if self.args.mode == 'player_vs_model':
                # Player vs Model mode
                p2_actions = self.display_env.player_actions
                # Combine actions properly for 2-player game
                actions = [p1_actions[0], p2_actions]
            else:
                # Model vs Game mode
                actions = [p1_actions[0]]

            self.display_env.action_probabilities = []

            for i in range(4):
                if self.need_display:
                    self.display_env.set_ai_sys_info(self.ai_sys)
                state, reward, done, info = self.display_env.step(actions)
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

    if args.mode != 'model_vs_model':
        games.wrappers.init(args)

    player = NHL94Player(args, logger)

    com_print('========= Start of Game Loop ==========')
    if args.mode == 'player_vs_model':
        com_print('Press ESC or Q to quit')

    player.play(need_reset=False)

if __name__ == '__main__':
    main(sys.argv)