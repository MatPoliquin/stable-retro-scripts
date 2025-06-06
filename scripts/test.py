import unittest
import subprocess
import sys
import os
import torch
import argparse

#========================================================
# Pytorch
#========================================================
class TestPyTorchInstallation(unittest.TestCase):
    def test_torch_import(self):
        """Test if torch module can be imported."""
        try:
            import torch
        except ImportError:
            self.fail("PyTorch is not installed or cannot be imported.")

    def test_torch_version(self):
        """Test if torch.version is accessible and returns a string."""
        version = torch.__version__
        self.assertIsInstance(version, str, "Torch version is not a string.")
        self.assertTrue(len(version) > 0, "Torch version string is empty.")

    def test_tensor_creation(self):
        """Test if we can create a tensor."""
        tensor = torch.tensor([1,2,3])
        self.assertTrue(torch.is_tensor(tensor), "Created object is not a tensor.")

    def test_tensor_operations(self):
        """Test basic tensor operations."""
        a = torch.tensor([1,2,3])
        b = torch.tensor([4,5,6])
        c = a + b
        expected = torch.tensor([5,7,9])
        self.assertTrue(torch.equal(c, expected), "Tensor addition result is incorrect.")

    def test_cuda_availability(self):
        """Check if CUDA is available or not."""
        cuda_available = torch.cuda.is_available()
        self.assertIsInstance(cuda_available, bool, "CUDA availability should be boolean.")

#========================================================
# Stable-baselines 3
#========================================================
class TestSB3(unittest.TestCase):

    def test_single_player(self):
        with open(os.devnull, 'w') as f:
            original_stdout = sys.stdout
            import retro
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv

            GAME_ENV = 'Airstriker-Genesis'
            STATE_1P = 'Level1'
            POLICY = 'CnnPolicy'
            TIMESTEPS = 1000

            # Create env and apply wrappers
            env = retro.make(game=GAME_ENV, state=STATE_1P, render_mode="rgb_array")
            env = WarpFrame(env)
            env = ClipRewardEnv(env)
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=4)

            # Train model
            model = PPO(policy=POLICY, env=env, verbose=False, n_epochs = 4, batch_size=32, ent_coef=0.01)
            model.learn(total_timesteps=TIMESTEPS)

            # Test the trained model
            state = env.reset()

            while True:
                #env.render(mode='human')

                actions = model.predict(state, deterministic=True)

                state, reward, done, info = env.step(actions[0])

                if done:
                    env.reset()
                    break

            env.close()
            sys.stdout = original_stdout

    def test_two_players(self):
        with open(os.devnull, 'w') as f:
            import retro
            import numpy as np
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
            from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv

            GAME_ENV = 'Pong-Atari2600'
            STATE_1P = 'Start'
            STATE_2P = 'Start.2P'
            POLICY = 'CnnPolicy'
            TIMESTEPS = 1000

            original_stdout = sys.stdout
            # Create Env
            env = retro.make(game=GAME_ENV, state=STATE_1P, render_mode="rgb_array") # Creates the env that contains the genesis emulator
            env = WarpFrame(env)                         # Downsamples the game frame buffer to 84x84 greyscale pixel
            env = ClipRewardEnv(env)                     # Make sure returned reward from env is not out of bounds
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=4)


            # Create p1 model that will be trained with PPO2 algo
            p1_model = PPO(policy=POLICY, env=env, verbose=False)
            # Train p1 model on env for X timesteps
            p1_model.learn(total_timesteps=TIMESTEPS)

            # Create p2 model that will be trained with PPO2 algo
            p2_model = PPO(policy=POLICY, env=env, verbose=False)
            # Train p2 model on env for X timesteps
            p2_model.learn(total_timesteps=TIMESTEPS)

            # Close previous env since we cannot have more than one in this same process
            env.close()

            # Create 2 player env
            env_2p = retro.make(game=GAME_ENV, state=STATE_2P, players=2, render_mode="rgb_array") # Creates the env that contains the genesis emulator
            env_2p = WarpFrame(env_2p)                         # Downsamples the game frame buffer to 84x84 greyscale pixel
            env_2p = ClipRewardEnv(env_2p)                     # Make sure returned reward from env is not out of bounds
            env_2p = DummyVecEnv([lambda: env_2p])
            env_2p = VecFrameStack(env_2p, n_stack=4)

            #env_2p = VecTransposeImage(env_2p)

            # Test the trained model
            state = env_2p.reset()

            while True:
                #env_2p.render()

                # model takes as input a stack of 4 x 84x84 frames
                # returns which buttons on the Genesis gamepad was pressed (an array of 12 bools)
                p1_actions = p1_model.predict(state)
                p2_actions = p2_model.predict(state)

                #actions = env_2p.unwrapped.action_space.sample()
                actions = np.append(p1_actions[0], p2_actions[0])
                #print(actions)

                # pass those actions to the environement (emulator) so it can generate the next frame
                # return:
                # state = next stack of image
                # reward outcome of the environement
                # done: if the game is over
                # info: variables used to create the reward and done functions (for debugging)
                state, reward, done, info = env_2p.step([actions])

                if done:
                    env_2p.reset()
                    break
            env_2p.close()
            sys.stdout = original_stdout

#========================================================
# Scripts
#========================================================
class TestScripts(unittest.TestCase):

    def test_trainer(self):
        command = [
            "python3",
            "model_trainer.py",
            "--env=Airstriker-Genesis",
            "--num_env=2",
            "--num_timesteps=1_000"
        ]

        # Run the command and check that it completes successfully
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self.fail(f"Command failed with return code {e.returncode}\nSTDERR: {e.stderr}")

        # Optionally, assert something about the output if needed:
        #self.assertIn("Expected output or status", result.stdout)

    def test_model_vs_game(self):
        command = [
            "python3",
            "model_vs_game.py",
            "--env=Airstriker-Genesis",
            "--num_timesteps=1_000",
            "--model_1=../models/airstriker_nature_cnn.zip"
        ]

        # Run the command and check that it completes successfully
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self.fail(f"Command failed with return code {e.returncode}\nSTDERR: {e.stderr}")

    def test_model_vs_model(self):
        command = [
            "python3",
            "model_vs_model.py",
            "--env=Pong-Atari2600",
            "--load_p1_model=../models/pong_nature_cnn.zip",
            "--load_p2_model=../models/pong_nature_cnn.zip"
        ]

        # Run the command and check that it completes successfully
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self.fail(f"Command failed with return code {e.returncode}\nSTDERR: {e.stderr}")
        return

#========================================================
# Models
#========================================================
class TestModels(unittest.TestCase):
    base_command = [
        "python3",
        "model_trainer.py",
        "--env=Pong-Atari2600",
        "--num_env=2",
        "--num_timesteps=1_000"
    ]

    def run_command(self, nn_policy):
        command = self.base_command + [f"--nn={nn_policy}"]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            self.fail(f"Command failed with return code {e.returncode}\nSTDERR: {e.stderr}")

    def test_policies(self):
        for policy in ["CnnPolicy", "CustomCnnPolicy", "CustomMlpPolicy", "MlpPolicy", "CombinedPolicy", "ImpalaCnnPolicy"]:
            with self.subTest(nn_policy=policy):
                result = self.run_command(policy)
                # Example assertion, modify as needed:
                # self.assertIn("Training completed", result.stdout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run unit tests with verbosity option.")
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        default=2,  # default verbosity = 2 (detailed)
        choices=[0, 1, 2],
        help="Set verbosity level for unittest (0 = quiet, 1 = default, 2 = detailed). Default is 2."
    )
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help="Specify a test or test pattern to run (e.g., TestSB3.test_single_player)"
    )
    args, unknown = parser.parse_known_args()

    # Build argv to pass to unittest.main()
    # Start with script name
    unittest_argv = [sys.argv[0]]

    if args.test:
        # Add the test name or pattern to argv for unittest
        unittest_argv.append(args.test)

    # Append any unknown args (like -v, etc. not parsed by argparse)
    unittest_argv.extend(unknown)

    # Run the unit tests and display report with user-defined verbosity
    unittest.main(verbosity=args.verbosity, argv=unittest_argv)