import os
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import retro
import game_wrappers_mgr as games
import cv2
from collections import deque

def isMLP(name):
    return name == 'MlpPolicy' or name == 'MlpDropoutPolicy' or name == 'CombinedPolicy' or name == 'AttentionMLPPolicy' or name == 'EntityAttentionPolicy'

class WarpFrameDict(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale

        img_space = env.observation_space.spaces['image']
        n_channels = 1 if self.grayscale else img_space.shape[0]

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(n_channels, self.height, self.width), dtype=np.uint8),
            'scalar': env.observation_space.spaces['scalar']
        })

    def observation(self, obs):
        img = obs['image']  # Expect shape: (C, H, W)

        # Check shape and type for debugging
        #print(f"Original img shape: {img.shape}, dtype: {img.dtype}")

        # Make sure img is a numpy array
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Validate shape has 3 dimensions
        assert len(img.shape) == 3, f"Expected image with 3 dims (C,H,W), got {img.shape}"

        # Convert CHW to HWC for OpenCV
        #img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        # Check shape after transpose
        #print(f"After transpose img shape: {img.shape}, dtype: {img.dtype}")
        #assert img.shape[2] in [1, 3, 4], f"Invalid channel number {img.shape[2]} for cv2"

        if self.grayscale:
            # Convert RGB to grayscale (input must have 3 or 4 channels)
            if img.shape[2] == 1:
                # Already grayscale, just resize
                img = img[:, :, 0]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Results in (H, W)

            # Resize grayscale
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # Add channel dimension back (1, H, W)
            img = np.expand_dims(img, axis=0)
        else:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            # Convert back to CHW
            #img = np.transpose(img, (2, 0, 1))

        img = img.astype(np.uint8)

        return {'image': img, 'scalar': obs['scalar']}

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, terminated, truncated, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated: break
        return ob, totrew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)

def make_retro(*, game, state=None, num_players, max_episode_steps=4500, **kwargs):
    import retro  # pylint: disable=import-outside-toplevel,reimported
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, players=num_players, render_mode="rgb_array")
    #env = NHL94Discretizer(env)
    #if max_episode_steps is not None:
    #    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def init_env(output_path, num_env, state, num_players, args, use_sticky_action=True, use_display=False, use_frame_skip=True):
    #if wrapper_kwargs is None:
    wrapper_kwargs = {}
    #wrapper_kwargs['scenario'] = 'test'

    seed = 0
    start_index = 0
    start_method=None
    allow_early_resets=True

    def make_env(rank):
        def _thunk():
            games.wrappers.init(args)

            env = make_retro(game=args.env, use_restricted_actions=retro.Actions.FILTERED, state=state, num_players=num_players)

            env.action_space.seed(seed + rank)

            # TOFIX
            #if use_display:
            #    env = GameDisplayEnv(env, args, 17, 'CNN', None)
            if use_frame_skip:
                if use_sticky_action:
                    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
                else:
                    env = StochasticFrameSkip(env, n=4, stickprob=-1)

            if isMLP(args.nn):
                env = games.wrappers.obs_env(env, args, num_players, args.rf)
                #if args.rf != '':
                #    env.set_reward_function(args.rf)

            env = Monitor(env, output_path and os.path.join(output_path, str(rank)), allow_early_resets=allow_early_resets)


            if not isMLP(args.nn):
                env = WarpFrame(env)
            elif args.nn == 'CombinedPolicy':
                env = WarpFrameDict(env)

            #if args.nn == 'EntityAttentionPolicy':
            #    env = EntityAttentionWrapper(env, num_frames=4)

            env = ClipRewardEnv(env)

            return env
        return _thunk

    env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], start_method=start_method)

    env.seed(seed)

    if not isMLP(args.nn) or args.nn == 'CombinedPolicy':
        env = VecFrameStack(env, n_stack=4)
        #env = VecTransposeImage(env)

    return env


def get_button_names(args):
    env = retro.make(game=args.env, state=args.state, use_restricted_actions=retro.Actions.FILTERED, players=args.num_players)
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
