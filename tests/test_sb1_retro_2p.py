import retro
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.atari_wrappers import WarpFrame, ClipRewardEnv, FrameStack

GAME_ENV = 'Pong-Atari2600'
STATE_1P = 'Start'
STATE_2P = 'Start.2P'
POLICY = 'CnnPolicy'
TIMESTEPS = 10000


def apply_wrappers(env):
    env = WarpFrame(env)                         # Downsamples the game frame buffer to 84x84 greyscale pixel
    env = FrameStack(env, 4)                     # Creates a stack of the last 4 frames to encode velocity
    env = ClipRewardEnv(env)                     # Make sure returned reward from env is not out of bounds

    return env

def main():
    # Create Env
    env = retro.make(game=GAME_ENV, state=STATE_1P) # Creates the env that contains the genesis emulator
    apply_wrappers(env)

    # Create p1 model that will be trained with PPO2 algo
    p1_model = PPO2(policy=POLICY, env=env, verbose=True)
    # Train p1 model on env for X timesteps
    p1_model.learn(total_timesteps=TIMESTEPS)

    # Create p2 model that will be trained with PPO2 algo
    p2_model = PPO2(policy=POLICY, env=env, verbose=True)
    # Train p2 model on env for X timesteps
    p2_model.learn(total_timesteps=TIMESTEPS)

    # Close previous env since we cannot have more than one in this same process
    env.close()

    # Create 2 player env
    env_2p = retro.make(game=GAME_ENV, state=STATE_2P, players=2) # Creates the env that contains the genesis emulator
    apply_wrappers(env_2p)

    # Test the trained model
    state = env_2p.reset()

    while True:
        env_2p.render()

        # model takes as input a stack of 4 x 84x84 frames
        # returns which buttons on the Genesis gamepad was pressed (an array of 12 bools)
        p1_actions = p1_model.predict(state)
        p2_actions = p2_model.predict(state)
        
        #actions = env_2p.unwrapped.action_space.sample()
        actions = np.append(p1_actions[0], p2_actions[0])

        # pass those actions to the environement (emulator) so it can generate the next frame
        # return:
        # state = next stack of image
        # reward outcome of the environement
        # done: if the game is over
        # info: variables used to create the reward and done functions (for debugging)
        state, reward, done, info = env_2p.step(actions)

        if done:
            env_2p.reset()

if __name__ == '__main__':
    main()