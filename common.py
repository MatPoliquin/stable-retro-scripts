"""
Common utils
"""

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

from gymnasium import spaces
from collections import deque

logger = None

def create_output_dir(args):
    output_dir = args.env + datetime.datetime.now().strftime('-%Y-%m-%d_%H-%M-%S')
    output_fullpath = os.path.join(os.path.expanduser(args.output_basedir), output_dir)
    os.makedirs(output_fullpath, exist_ok=True)
    
    #logger.configure(output_fullpath)

    return output_fullpath

def init_logger(args):
    tmp_path = create_output_dir(args)
    #tmp_path = "/tmp/sb3_log/"
    # set up logger
    global logger
    logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    logger.log("TEST!!!!!!!!!!!!!!!!!!!!!")

    print(logger)

    return logger

def com_print(text):
    logger.log(text)

def get_model_file_name(args):
    return args.env + '-' + args.alg + '-' + args.nn + '-' + str(args.num_timesteps)



