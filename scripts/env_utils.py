import os
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import stable_retro as retro
import stable_retro.data
import game_wrappers_mgr as games
import cv2
from env_wrappers import StochasticFrameSkip, WarpFrameDict, RewardClipper
from utils import resolve_clip_reward, resolve_sticky_action_settings


def isMLP(name):
    return name == 'MlpPolicy' or name == 'MlpDropoutPolicy' or name == 'CombinedPolicy' \
          or name == 'AttentionMLPPolicy' or name == 'EntityAttentionPolicy' or name == 'HockeyMultiHeadPolicy' \
          or name == 'ResidualMlpPolicy' \
          or name == 'HybridMambaPolicy' or name == 'GRUMlpPolicy' or name == 'ClassicAI'


def resolve_backend_action_type(args, num_players):
    requested_action_type = getattr(args, 'action_type', 'FILTERED').upper()
    if (
        getattr(args, 'env', '') == 'MortalKombatII-Genesis-v0'
        and isMLP(getattr(args, 'nn', ''))
        and num_players == 1
        and requested_action_type in ('DISCRETE', 'MULTI_DISCRETE')
    ):
        return 'FILTERED'
    return requested_action_type


def resolve_retro_state_name(game, state, num_players, inttype):
    if num_players <= 1:
        return state

    if state in (None, retro.State.DEFAULT, retro.State.NONE):
        return state

    if not isinstance(state, str):
        return state

    base_state = state[:-6] if state.endswith('.state') else state
    if base_state.endswith('.2P'):
        return base_state

    candidate = f"{base_state}.2P"
    if stable_retro.data.get_file_path(game, f"{candidate}.state", inttype):
        return candidate

    return base_state


def make_retro(
    *,
    game,
    state=None,
    num_players,
    max_episode_steps=4500,
    action_type='FILTERED',
    inttype=retro.data.Integrations.ALL,
    **kwargs,
):
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
    state = resolve_retro_state_name(game, state, num_players, inttype)

    env = retro.make(
        game,
        state,
        **kwargs,
        players=num_players,
        render_mode="rgb_array",
        use_restricted_actions=action_enum,
        inttype=inttype,
    )
    #env = NHL94Discretizer(env)
    #if max_episode_steps is not None:
    #    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def init_env(
    output_path,
    num_env,
    state,
    num_players,
    args,
    hyperparams,
    use_sticky_action=True,
    use_display=False,
    use_frame_skip=True,
):
    wrapper_kwargs = {}

    if getattr(args, 'selfplay', False) and not isMLP(args.nn):
        raise ValueError('Self-play currently requires an MLP-style observation wrapper.')

    clip_reward = resolve_clip_reward(args, hyperparams)
    args.clip_reward = clip_reward
    sticky_actions_enabled, sticky_action_prob = resolve_sticky_action_settings(
        use_sticky_action,
        hyperparams,
    )
    frame_skip = max(1, int(hyperparams.get('frame_skip', 4)))

    seed = 0
    start_index = 0
    start_method = os.environ.get('RETRO_VECENV_START_METHOD')
    allow_early_resets=True
    env_num_players = 2 if getattr(args, 'selfplay', False) else num_players

    # N64 cores/plugins are commonly unsafe under Linux's default multiprocessing start method (fork).
    # Prefer spawn unless the user explicitly overrides via RETRO_VECENV_START_METHOD.
    if start_method is None and getattr(args, 'selfplay', False):
        start_method = 'spawn'
    elif start_method is None and hasattr(args, 'env') and args.env and 'N64' in args.env:
        start_method = 'spawn'

    def make_env(rank):
        def _thunk():
            games.wrappers.init(args)

            backend_action_type = resolve_backend_action_type(args, env_num_players)
            env = make_retro(game=args.env, action_type=backend_action_type, state=state, num_players=env_num_players)

            env.action_space.seed(seed + rank)

            if isMLP(args.nn):
                env = games.wrappers.obs_env(env, args, env_num_players, args.rf)

            env = Monitor(env, output_path and os.path.join(output_path, str(rank)), allow_early_resets=allow_early_resets)

            if use_frame_skip:
                if sticky_actions_enabled:
                    env = StochasticFrameSkip(env, n=frame_skip, stickprob=sticky_action_prob)
                else:
                    env = StochasticFrameSkip(env, n=frame_skip, stickprob=-1)

            if not isMLP(args.nn):
                env = WarpFrame(env)
            elif args.nn == 'CombinedPolicy':
                env = WarpFrameDict(env)

            if clip_reward:
                env = RewardClipper(env, low=-1.0, high=1.0)

            return env
        return _thunk

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
    backend_action_type = resolve_backend_action_type(args, 2 if getattr(args, 'selfplay', False) else args.num_players)
    action_enum = action_map.get(backend_action_type.upper(), retro.Actions.FILTERED)

    env = retro.make(
        game=args.env,
        state=args.state,
        use_restricted_actions=action_enum,
        players=2 if getattr(args, 'selfplay', False) else args.num_players,
        inttype=retro.data.Integrations.ALL,
    )
    try:
        games.wrappers.init(args)
        if isMLP(args.nn) and games.wrappers.obs_env is not None:
            wrapped_env = games.wrappers.obs_env(
                env,
                args,
                2 if getattr(args, 'selfplay', False) else args.num_players,
                getattr(args, 'rf', ''),
            )
            if hasattr(wrapped_env, 'get_action_names'):
                return wrapped_env.get_action_names()
        return env.buttons
    finally:
        env.close()

def init_play_env(args, num_players, hyperparams, is_pvp_display=False, need_display=True, use_frame_skip=True):
    button_names = get_button_names(args)

    env = init_env(
        None,
        1,
        args.state,
        num_players,
        args,
        hyperparams,
        use_sticky_action=False,
        use_display=False,
        use_frame_skip=use_frame_skip,
    )

    if not need_display:
        return env

    games.wrappers.init(args)

    if is_pvp_display:
        display_env = env = games.wrappers.pvp_display_env(env, args, args.model1_desc, args.model2_desc, None, None, button_names)
    else:
        display_env = env = games.wrappers.sp_display_env(env, args, 0, args.model1_desc, button_names)

    return display_env
