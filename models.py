import warnings
warnings.filterwarnings("ignore")
import os, datetime
import argparse
import retro
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import FrameStack


import torch as th
from torchsummary import summary

from stable_baselines3.common.logger import configure


def print_model_info(model):
    total_params = 0
    for v in model.get_parameter_list():
        logger.log(v)
        shape = v.get_shape()
        count = 1
        for dim in shape:
            count *= dim.value
            total_params += count
    logger.log("Total trainable parameters:%d" % total_params)
    logger.log("=========            =========")

    return total_params

def get_num_parameters(model):
    #total_params = 0
    #for v in model.get_parameters():
    #    shape = v.get_shape()
    #    count = 1
    #    for dim in shape:
    #        count *= dim.value
    #        total_params += count

    total_params = 100000000
    return total_params

def get_model_probabilities(model, state):
    #obs = obs_as_tensor(state, model.policy.device)
    obs = model.policy.obs_to_tensor(state)[0]
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np

def init_model(output_path, player_model, player_alg, args, env, logger):

    policy_kwargs=None
    if args.nn == 'MlpPolicy':
        size = args.nnsize
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[size, size], vf=[size, size])])

    if player_alg == 'ppo2':
        if player_model == '':
            batch_size = (128 * args.num_env) // 4
            print("batch_size:")
            print(batch_size)
            batch_size = 64
            model = PPO(policy=args.nn, env=env, policy_kwargs=policy_kwargs, verbose=1, n_steps = 2048, n_epochs = 4, batch_size = batch_size, learning_rate = 2.5e-4, clip_range = 0.2, vf_coef = 0.5, ent_coef = 0.01,
                 max_grad_norm=0.5, clip_range_vf=None)
        else:
            model = PPO.load(os.path.expanduser(player_model), env=env)
    elif player_alg == 'a2c':
        if player_model == '':
            model = A2C(policy=args.nn, env=env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            model = A2C(policy=args.nn, env=env, verbose=1, tensorboard_log=output_path)

    model.set_logger(logger)


    print(model.policy)
    
    # TODO Get model input size
    #if args.nn == 'MlpPolicy':
    #    summary(model.policy, (1, 6))

    return model