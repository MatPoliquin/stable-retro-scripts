"""
Common utils
"""

import os, datetime
import argparse
import retro
import gym
import numpy as np
import pygame
from stable_baselines import PPO2, A2C
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.atari_wrappers import WarpFrame, ClipRewardEnv, FrameStack
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines import logger
from baselines.common.retro_wrappers import StochasticFrameSkip

# GameDisplay
class GameDisplay:
    def __init__(self, args):
        # Init Window
        pygame.init()
        #screen = pygame.display.set_mode((1920, 1080))
        self.screen = pygame.display.set_mode((960, 540))
        self.main_surf = pygame.Surface((1920, 1080))
        self.main_surf.set_colorkey((0,0,0))


    def draw_frame(self, frame_img):
        self.main_surf.fill((0, 0, 0))
        emu_screen = np.transpose(frame_img, (1,0,2))
        #print(emu_screen.shape)
        #surf.fill((0,0,0))
        surf = pygame.surfarray.make_surface(emu_screen)

        self.main_surf.blit(pygame.transform.scale(surf,(1920,1080)), (0, 0))
        #print(main_surf.get_colorkey())
        self.main_surf.set_colorkey(None)
        #main_surf.convert()
        self.screen.blit(pygame.transform.smoothscale(self.main_surf,(960,540)), (0, 0))
        #screen.blit(surf, (0, 0))
 
        pygame.display.flip()

    def GetInput(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        return keystate

def make_retro(*, game, state=None, num_players, max_episode_steps=4500, **kwargs):
    import retro
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, players=num_players)
    #if max_episode_steps is not None:
    #    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def init_env(output_path, num_env, state, num_players, args):
    #if wrapper_kwargs is None:
    wrapper_kwargs = {}
    #wrapper_kwargs['scenario'] = 'test'

    seed = 0
    start_index = 0
    start_method=None
    allow_early_resets=True

    def make_env(rank):
        def _thunk():
            env = make_retro(game=args.env, use_restricted_actions=retro.Actions.FILTERED, state=state, num_players=num_players)

            env.seed(seed + rank)
            #print(logger.get_dir())
            env = Monitor(env, output_path and os.path.join(output_path, str(rank)), allow_early_resets=allow_early_resets)

            env = WarpFrame(env)
            env = ClipRewardEnv(env)
            env = StochasticFrameSkip(env, n=4, stickprob=0.25)

            return env
        return _thunk
    set_global_seeds(seed)


    env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], start_method=start_method)

    env = VecFrameStack(env, n_stack=4)

    print(env.action_space)

    return env

def init_play_env(args):
    env = retro.make(game=args.env, state=args.state, use_restricted_actions=retro.Actions.FILTERED, players=args.num_players)

    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)

    return env


def init_model(output_path, player_model, player_alg, args, env):
    if player_alg == 'ppo2':
        if player_model == '':
            model = PPO2(policy=args.nn, env=env, verbose=args.alg_verbose, tensorboard_log=output_path)
        else:
            model = PPO2.load(os.path.expanduser(player_model), env=env, policy=CnnPolicy)
    elif player_alg == 'a2c':
        if player_model == '':
            model = A2C(policy=args.nn, env=env, verbose=1, tensorboard_log=output_path)
        else:
            model = A2C(policy=args.nn, env=env, verbose=1, tensorboard_log=output_path)

    return model

def create_output_dir(args):
    output_dir = args.env + datetime.datetime.now().strftime('-%Y-%m-%d_%H-%M-%S')
    output_fullpath = os.path.join(os.path.expanduser(args.output_basedir), output_dir)
    os.makedirs(output_fullpath, exist_ok=True)
    
    logger.configure(output_fullpath)

    return output_fullpath

def get_model_file_name(args):
    return args.env + '-' + args.alg + '-' + args.nn + '-' + str(args.num_timesteps)

