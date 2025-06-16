# es.py
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # 3 actions: NOOP, UP, DOWN

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

NUM_PARAMS = 6 # Hack, need to manualy set obs size for now, should get this from env
class EvolutionStrategies:
    def __init__(self, env, args, num_players, rf_name):
        self.env = env
        self.args = args
        self.obs_size = NUM_PARAMS if args.nn != 'CombinedPolicy' else NUM_PARAMS

        # ES parameters
        self.population_size = 50
        self.sigma = 0.1
        self.learning_rate = 0.01
        self.decay = 0.995

        # Policy network
        self.policy = PolicyNetwork(self.obs_size)
        self.weights = self.get_weights()

        # Tracking
        self.reward_history = deque(maxlen=100)
        self.best_reward = -float('inf')

        # Get action space info from environment
        self.action_size = env.action_space.n  # Number of discrete actions

    def get_weights(self):
        return torch.cat([param.data.view(-1) for param in self.policy.parameters()])

    def set_weights(self, weights):
        idx = 0
        for param in self.policy.parameters():
            size = param.data.numel()
            param.data = weights[idx:idx+size].view(param.data.shape)
            idx += size

    def get_action(self, obs):
        """Convert policy output to environment-compatible action"""
        if self.args.nn == 'CombinedPolicy':
            obs_tensor = torch.FloatTensor(obs['scalar'])
        else:
            obs_tensor = torch.FloatTensor(obs)

        with torch.no_grad():
            action_probs = self.policy(obs_tensor)

        #action_idx = torch.multinomial(action_probs, 1).item()
        #return action_idx  # Return the discrete action index

        return action_probs

    def evaluate(self, weights=None, render=False, num_episodes=1):
        if weights is not None:
            self.set_weights(weights)

        total_reward = 0.0

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False

            while not done:
                action = self.get_action(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += float(reward)

                if render:
                    self.env.render()

        return float(total_reward / num_episodes)

    def train(self, num_generations=1000):
        best_weights = self.weights.clone()

        for generation in range(num_generations):
            # Generate population
            population = []
            for _ in range(self.population_size):
                noise = torch.randn_like(self.weights) * self.sigma
                population.append((self.weights + noise, noise))

            # Evaluate population
            rewards = np.zeros(self.population_size)
            for i, (candidate, _) in enumerate(population):
                rewards[i] = self.evaluate(candidate)

            # Update weights
            norm_rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            update = torch.zeros_like(self.weights)

            for i in range(self.population_size):
                _, noise = population[i]
                update += norm_rewards[i] * noise

            self.weights += self.learning_rate * update / self.population_size
            self.set_weights(self.weights)

            # Track progress
            current_reward = self.evaluate()
            self.reward_history.append(current_reward)

            if current_reward > self.best_reward:
                self.best_reward = current_reward
                best_weights = self.weights.clone()

            # Decay and log
            self.learning_rate *= self.decay
            if generation % 10 == 0:
                print(f"Gen {generation}: Reward {current_reward:.2f} (Best: {self.best_reward:.2f})")

        self.set_weights(best_weights)

    def save_model(self, path):
        torch.save({
            'weights': self.weights,
            'policy_state_dict': self.policy.state_dict(),
            'best_reward': self.best_reward
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.weights = checkpoint['weights']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.best_reward = checkpoint['best_reward']