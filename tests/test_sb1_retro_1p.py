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
    env = retro.make(game=GAME_ENV, state=STATE) # Creates the env that contains the genesis emulator
    env = WarpFrame(env)                         # Downsamples the game frame buffer to 84x84 greyscale pixel
    env = FrameStack(env, 4)                     # Creates a stack of the last 4 frames to encode velocity
    env = ClipRewardEnv(env)                     # Make sure returned reward from env is not out of bounds

    # Create model that will be trained with PPO2 algo
    model = PPO2(policy=POLICY, env=env, verbose=True)

    # Train model on env for X timesteps
    model.learn(total_timesteps=TIMESTEPS)

    # Test the trained model
    state = env.reset()

    while True:
        env.render()

        # model takes as input a stack of 4 x 84x84 frames
        # returns which buttons on the Genesis gamepad was pressed (an array of 12 bools)
        actions = model.predict(state)

        # pass those actions to the environement (emulator) so it can generate the next frame
        # return:
        # state = next stack of image
        # reward outcome of the environement
        # done: if the game is over
        # info: variables used to create the reward and done functions (for debugging)
        state, reward, done, info = env.step(actions[0])

        if done:
            env.reset()

if __name__ == '__main__':
    main()