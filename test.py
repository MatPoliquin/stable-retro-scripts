import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
import gymnasium as gym

class CustomMLPExtractor(nn.Module):
    def __init__(self, feature_dim, net_arch, dropout_prob=0.5):
        super().__init__()

        # Shared layers with BatchNorm and Dropout
        shared_layers = []
        last_layer_dim = feature_dim
        for layer_size in net_arch:
            shared_layers.append(nn.Linear(last_layer_dim, layer_size))
            shared_layers.append(nn.BatchNorm1d(layer_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_prob))
            last_layer_dim = layer_size
        self.shared_net = nn.Sequential(*shared_layers)

        # Policy head with BatchNorm and Dropout
        self.policy_net = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim),
            nn.BatchNorm1d(last_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        # Value head with BatchNorm and Dropout
        self.value_net = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim),
            nn.BatchNorm1d(last_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        shared = self.shared_net(features)
        return self.policy_net(shared)

    def forward_critic(self, features):
        shared = self.shared_net(features)
        return self.value_net(shared)

class CustomDropoutPolicy(ActorCriticPolicy):
    def __init__(self, *args, dropout_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        net_arch = kwargs.get("net_arch", [64, 64])
        self.mlp_extractor = CustomMLPExtractor(self.features_dim, net_arch, dropout_prob)

def main():
    env = gym.make("CartPole-v1", render_mode="human")

    model = PPO(
        policy=CustomDropoutPolicy,
        env=env,
        verbose=1,
        policy_kwargs=dict(net_arch=[64, 64], dropout_prob=0.3)
    )

    model.learn(total_timesteps=10000)

    model.save("ppo_cartpole_custom_policy")

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()  # This will now work properly
        if dones:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()