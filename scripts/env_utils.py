import os
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import stable_retro as retro
import game_wrappers_mgr as games
import cv2
from env_wrappers import StochasticFrameSkip, WarpFrameDict

def isMLP(name):
    return name == 'MlpPolicy' or name == 'MlpDropoutPolicy' or name == 'CombinedPolicy' \
          or name == 'AttentionMLPPolicy' or name == 'EntityAttentionPolicy' or name == 'HockeyMultiHeadPolicy'


def make_retro(*, game, state=None, num_players, max_episode_steps=4500, action_type='FILTERED', **kwargs):
    import stable_retro as retro  # pylint: disable=import-outside-toplevel,reimported
    if state is None:
        state = retro.State.DEFAULT

    # Convert action_type string to retro.Actions enum
    action_map = {
        'FILTERED': retro.Actions.FILTERED,
        'DISCRETE': retro.Actions.DISCRETE,
        'MULTI_DISCRETE': retro.Actions.MULTI_DISCRETE
    }
    action_enum = action_map.get(action_type.upper(), retro.Actions.FILTERED)

    env = retro.make(game, state, **kwargs, players=num_players, render_mode="rgb_array", use_restricted_actions=action_enum)
    #env = NHL94Discretizer(env)
    #if max_episode_steps is not None:
    #    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def init_env(output_path, num_env, state, num_players, args, use_sticky_action=True, use_display=False, use_frame_skip=True):
    wrapper_kwargs = {}

    seed = 0
    start_index = 0
    start_method = os.environ.get('RETRO_VECENV_START_METHOD')
    allow_early_resets=True

    # N64 cores/plugins are commonly unsafe under Linux's default multiprocessing start method (fork).
    # Prefer spawn unless the user explicitly overrides via RETRO_VECENV_START_METHOD.
    if start_method is None and hasattr(args, 'env') and args.env and 'N64' in args.env:
        start_method = 'spawn'

    def make_env(rank):
        def _thunk():
            games.wrappers.init(args)

            env = make_retro(game=args.env, action_type=args.action_type, state=state, num_players=num_players)

            env.action_space.seed(seed + rank)

            if isMLP(args.nn):
                env = games.wrappers.obs_env(env, args, num_players, args.rf)

            env = Monitor(env, output_path and os.path.join(output_path, str(rank)), allow_early_resets=allow_early_resets)

            if use_frame_skip:
                if use_sticky_action:
                    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
                else:
                    env = StochasticFrameSkip(env, n=4, stickprob=-1)

            if not isMLP(args.nn):
                env = WarpFrame(env)
            elif args.nn == 'CombinedPolicy':
                env = WarpFrameDict(env)

            env = ClipRewardEnv(env)

            return env
        return _thunk

    if num_env == 1:
        env = DummyVecEnv([make_env(start_index)])
    else:
        env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], start_method=start_method)

    env.seed(seed)

    if not isMLP(args.nn) or args.nn == 'CombinedPolicy':
        env = VecFrameStack(env, n_stack=4)
        #env = VecTransposeImage(env)

    return env


def get_button_names(args):
    action_map = {
        'FILTERED': retro.Actions.FILTERED,
        'DISCRETE': retro.Actions.DISCRETE,
        'MULTI_DISCRETE': retro.Actions.MULTI_DISCRETE
    }
    action_enum = action_map.get(args.action_type.upper(), retro.Actions.FILTERED)

    env = retro.make(game=args.env, state=args.state, use_restricted_actions=action_enum, players=args.num_players)
    print(env.buttons)
    return env.buttons

def init_play_env(args, num_players, is_pvp_display=False, need_display=True, use_frame_skip=True):
    button_names = get_button_names(args)

    env = init_env(None, 1, args.state, num_players, args, use_sticky_action=False, use_display=False, use_frame_skip=use_frame_skip)

    if not need_display:
        return env

    games.wrappers.init(args)

    if is_pvp_display:
        display_env = env = games.wrappers.pvp_display_env(env, args, args.model1_desc, args.model2_desc, None, None, button_names)
    else:
        display_env = env = games.wrappers.sp_display_env(env, args, 0, args.model1_desc, button_names)

    return display_env
