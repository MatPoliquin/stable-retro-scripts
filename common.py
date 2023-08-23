"""
Common utils
"""

import warnings
warnings.filterwarnings("ignore")
import os, datetime
import argparse

from stable_baselines3.common.logger import configure

logger = None

def create_output_dir(args):
    output_dir = args.env + datetime.datetime.now().strftime('-%Y-%m-%d_%H-%M-%S')
    output_fullpath = os.path.join(os.path.expanduser(args.output_basedir), output_dir)
    os.makedirs(output_fullpath, exist_ok=True)
    
    #logger.configure(output_fullpath)

    return output_fullpath

def init_logger(args):
    tmp_path = create_output_dir(args)
    # set up logger
    global logger
    logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    return logger

def com_print(text):
    logger.log(text)

def get_model_file_name(args):
    return args.env + '-' + args.alg + '-' + args.nn + '-' + str(args.num_timesteps)
