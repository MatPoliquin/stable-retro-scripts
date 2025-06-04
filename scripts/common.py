"""
Common utils
"""


import os
import datetime

from stable_baselines3.common.logger import configure

LOGGER = None

def create_output_dir(args):
    """create_output_dir"""
    output_dir = args.env + datetime.datetime.now().strftime('-%Y-%m-%d_%H-%M-%S')
    output_fullpath = os.path.join(os.path.expanduser(args.output_basedir), output_dir)
    os.makedirs(output_fullpath, exist_ok=True)

    #logger.configure(output_fullpath)

    return output_fullpath

def init_logger(args):
    """init_logger"""
    tmp_path = create_output_dir(args)
    # set up logger
    global LOGGER
    LOGGER = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    return LOGGER

def com_print(text):
    """com_print"""
    LOGGER.log(text)

def get_model_file_name(args):
    """get_model_file_name"""
    return args.env + '-' + args.alg + '-' + args.nn + '-' + str(args.num_timesteps)
