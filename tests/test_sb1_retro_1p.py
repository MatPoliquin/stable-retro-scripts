import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
import gymnasium as gym

GAME_ENV = 'Pong-Atari2600'
STATE_1P = 'Start'
POLICY = 'CnnPolicy'
TIMESTEPS = 1000

def main():
    # Create env and apply wrappers
    env = retro.make(game=GAME_ENV, state=STATE_1P, render_mode="rgb_array")
    env = WarpFrame(env)
    env = ClipRewardEnv(env) 
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)    

    # Train model
    model = PPO(policy=POLICY, env=env, verbose=True)
    model.learn(total_timesteps=TIMESTEPS)

    # Test the trained model
    state = env.reset()

    while True:
        env.render(mode='human')

        actions = model.predict(state, deterministic=True)
   
        state, reward, done, info = env.step(actions[0])

        if done:
            env.reset()

if __name__ == '__main__':
    main()
