"""
Bare bones example to test stable-baselines setup
Airstriker-Genesis is a free game that comes with retro so no need to import any roms
"""

import retro
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.atari_wrappers import WarpFrame, ClipRewardEnv, FrameStack


GAME_ENV = 'Airstriker-Genesis'
STATE = 'Level1'
POLICY = 'CnnPolicy'
TIMESTEPS = 10000

def main():

    # Create Env
    env = retro.make(game=GAME_ENV, state=STATE)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = ClipRewardEnv(env)

    # Create Model
    model = PPO2(policy=POLICY, env=env, verbose=True)

    # Train model on env
    model.learn(total_timesteps=TIMESTEPS)

    # Test trained model
    state = env.reset()

    while True:
        env.render()

        actions = model.predict(state)

        state, reward, done, info = env.step(actions[0])

        if done:
            env.reset()


if __name__ == '__main__':
    main()