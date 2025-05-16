import os
from stable_baselines3 import PPO, A2C
import torch as th
import torch.nn as nn
from torchsummary import summary
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy



# ==========================================================================================
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Assuming the observation space is images with shape (C, H, W)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# ==========================================================================================
class CustomMLPExtractor(nn.Module):
    def __init__(self, feature_dim, net_arch, dropout_prob=0.5):
        super().__init__()

        # Shared layers
        shared_layers = []
        last_layer_dim = feature_dim
        for layer_size in net_arch:
            shared_layers.append(nn.Linear(last_layer_dim, layer_size))
            shared_layers.append(nn.BatchNorm1d(layer_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_prob))
            last_layer_dim = layer_size
        self.shared_net = nn.Sequential(*shared_layers)

        # Separate heads for policy and value functions
        self.policy_net = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim),
            nn.BatchNorm1d(last_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.value_net = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim),
            nn.BatchNorm1d(last_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, features):
        # Return policy and value features for the base policy's use
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

# ==========================================================================================
# Warning: input size is hardcoded for now
def print_model_info(model):

     if args.nn == 'CnnPolicy':
         summary(model.policy, (4, 84, 84))
     elif args.nn == 'MlpPolicy':
         summary(model.policy, (1, 6))

     return total_params

def get_num_parameters(model):
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    return total_params

def get_model_probabilities(model, state):
    #obs = obs_as_tensor(state, model.policy.device)
    obs = model.policy.obs_to_tensor(state)[0]
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np

def init_model(output_path, player_model, player_alg, args, env, logger):
    policy_kwargs=None
    nn_type = args.nn
    if args.nn == 'MlpPolicy':
        size = args.nnsize
        nn_type = 'MlpPolicy'
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[size, size], vf=[size, size]))
    elif args.nn == 'MlpDropoutPolicy':
        size = args.nnsize
        nn_type = CustomDropoutPolicy
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[size, size], dropout_prob=0.3)
    elif args.nn == 'CustomCnnPolicy':
        #size = args.nnsize
        nn_type = 'CnnPolicy'
        policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=128),)

    if player_alg == 'ppo2':
        if player_model == '':
            batch_size = (128 * args.num_env) // 4
            print("batch_size:%d" % batch_size)
            model = PPO(policy=nn_type, env=env, policy_kwargs=policy_kwargs, verbose=1, n_steps = 2048, n_epochs = 4, batch_size = batch_size, learning_rate = 2.5e-4, clip_range = 0.2, vf_coef = 0.5, ent_coef = 0.01,
                 max_grad_norm=0.5, clip_range_vf=None)
        else:
            model = PPO.load(os.path.expanduser(player_model), env=env)
    elif player_alg == 'a2c':
        if player_model == '':
            model = A2C(policy=nn_type, env=env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            model = A2C(policy=nn_type, env=env, verbose=1, tensorboard_log=output_path)

    model.set_logger(logger)

    #print_model_info(model)

    return model
