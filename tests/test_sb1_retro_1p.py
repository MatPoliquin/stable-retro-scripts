import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
import gymnasium as gym

GAME_ENV = 'Airstriker-Genesis'
STATE_1P = 'Level1'
POLICY = 'CnnPolicy'
TIMESTEPS = 10000000

def main():
    # Create env and apply wrappers
    env = retro.make(game=GAME_ENV, state=STATE_1P, render_mode="rgb_array")
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)    

    # Train model
    model = PPO(policy=POLICY, env=env, verbose=True, n_epochs = 4, batch_size=32, ent_coef=0.01)
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
