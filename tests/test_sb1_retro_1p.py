import retro
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from collections import deque
from gymnasium import spaces
import numpy as np
import os, datetime
import pygame

#GAME_ENV = 'Airstriker-Genesis'
GAME_ENV = 'Pong-Atari2600'
#STATE = 'Level1'
STATE = 'Start'
POLICY = 'CnnPolicy'
NUM_ENV = 8
TIMESTEPS = 10000
OUTPUT_DIR = '~/'

FB_WIDTH = 1920
FB_HEIGHT = 1080

#Taken from OpenAI baselines
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
    import retro
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, players=num_players)

    return env


def init_env(output_path, num_env, state, num_players, envid, use_frameskip=True, use_display=False):

    wrapper_kwargs = {}

    seed = 0
    start_index = 0
    start_method=None
    allow_early_resets=True
    
    def make_env(rank):
        def _thunk():
            env = make_retro(game=envid, use_restricted_actions=retro.Actions.DISCRETE, state=state, num_players=num_players, render_mode='rgb_array')

            env.action_space.seed(seed + rank)

            env = Monitor(env, output_path and os.path.join(output_path, str(rank)), allow_early_resets=allow_early_resets)

            if use_frameskip:
                env = StochasticFrameSkip(env, n=4, stickprob=-1)

            env = WarpFrame(env)
            
            env = ClipRewardEnv(env)

            return env
        return _thunk


    env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], start_method=start_method)

    env.seed(seed)
    
    env = VecFrameStack(env, n_stack=4)
    #env = VecTransposeImage(env)

    #print(env.action_space)

    return env

def init_play_env():
    env = retro.make(game=GAME_ENV, state=STATE)
    env = StochasticFrameSkip(env, n=4, stickprob=-1)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, num_stack=4)

    return env

class FullScreenDisplayEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.GAME_WIDTH = int(FB_WIDTH * 0.8)
        self.GAME_HEIGHT = FB_HEIGHT
        
        # Init Window
        pygame.init()
        self.screen = pygame.display.set_mode((FB_WIDTH, FB_HEIGHT), pygame.FULLSCREEN | pygame.NOFRAME | pygame.SCALED, vsync=1)
        self.main_surf = pygame.Surface((FB_WIDTH, FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, ac):
        ob, rew, done, info = self.env.step(ac)

        framebuffer = self.env.render(mode='rgb_array')

        self.main_surf.fill((0, 0, 0))
        emu_screen = np.transpose(framebuffer, (1,0,2))

        surf = pygame.surfarray.make_surface(emu_screen)

        self.main_surf.set_colorkey(None)
        x_pos = (FB_WIDTH - self.GAME_WIDTH) / 2
        self.main_surf.blit(pygame.transform.scale(surf,(self.GAME_WIDTH, self.GAME_HEIGHT)), (x_pos, 0))
        
        self.screen.blit(pygame.transform.smoothscale(self.main_surf,(FB_WIDTH, FB_HEIGHT)), (0, 0))
   
        pygame.display.flip()

        self.get_input()

        keystate = self.get_input()
        if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
            #logger.log('Exiting...')
            exit()
       
        return ob, rew, done, info

    def seed(self, s):
        self.rng.seed(s)

    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        return keystate

# ======================================================================
#   MAIN
# ======================================================================
def main():
    # Create Env
    #env = retro.make(game=GAME_ENV, state=STATE) # Creates the env that contains the genesis emulator
    #env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
    env = init_env(OUTPUT_DIR, NUM_ENV, STATE, 1, GAME_ENV)

    # Create model that will be trained with PPO2 algo
    model = PPO(policy=POLICY, env=env,
                learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1)

    # Train model on env for X timesteps
    model.learn(total_timesteps=TIMESTEPS)

    
    model.save("~/model.zip")

    del model

    play_env = init_env(OUTPUT_DIR, num_env=1, state=STATE, num_players=1, envid=GAME_ENV)


    model = PPO.load("~/model.zip", play_env)

    play_env = FullScreenDisplayEnv(model.get_env())


    #play_env = init_play_env() # Creates the env that contains the genesis emulator
    #model.set_env(play_env)

    # Test the trained model
    #play_env = init_play_env()

    state = play_env.reset()
    


    while True:
        #play_env.render(mode='human')

        # model takes as input a stack of 4 x 84x84 frames
        # returns which buttons on the Genesis gamepad was pressed (an array of 12 bools)
        #print(state)
        actions = model.predict(state)

        #print(actions)

        # pass those actions to the environement (emulator) so it can generate the next frame
        # return:
        # state = next stack of image
        # reward outcome of the environement
        # done: if the game is over
        # info: variables used to create the reward and done functions (for debugging)
        state, rew, terminated, info = play_env.step(actions[0])

        if terminated.all():
            play_env.reset()

if __name__ == '__main__':
    main()