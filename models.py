import os
from stable_baselines3 import PPO, A2C
import torch as th
import torch.nn as nn
from torchsummary import summary
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym


# ==========================================================================================
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim=128):
        # Extract the observation space dict components
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        # Assume observation_space is gym.spaces.Dict with keys "image" and "scalar"
        # Customize according to the actual keys and shapes in your observations

        # CNN for image input
        cnn_input_shape = observation_space.spaces['image'].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass with dummy data
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.spaces['image'].sample()[None]).float()).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )

        # For scalar inputs, fully connected layer
        scalar_dim = observation_space.spaces['scalar'].shape[0]
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU()
        )

        # Final combined features dimension
        self._features_dim = cnn_output_dim + 64

    def forward(self, observations):
        # observations is a dict with 'image' and 'scalar'
        img_tensor = observations['image']
        scalar_tensor = observations['scalar']

        cnn_features = self.cnn(img_tensor)
        cnn_features = self.cnn_fc(cnn_features)

        scalar_features = self.scalar_fc(scalar_tensor)

        # Concatenate features
        combined = th.cat([cnn_features, scalar_features], dim=1)
        return combined

# Use this custom extractor in a custom policy

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=128)
        )


# ==========================================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class ImpalaCNN(nn.Module):
    def __init__(self, input_channels=3, channels=(16, 32, 32)):
        super(ImpalaCNN, self).__init__()
        self.stacks = nn.ModuleList()
        in_channels = input_channels
        for out_channels in channels:
            self.stacks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    ResidualBlock(out_channels),
                    ResidualBlock(out_channels),
                )
            )
            in_channels = out_channels
        # Flatten feature size will depend on input image size. You may add an adaptive pooling here if needed.

    def forward(self, x):
        for stack in self.stacks:
            x = stack(x)
        x = th.flatten(x, start_dim=1)
        return x

class CustomImpalaFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomImpalaFeatureExtractor, self).__init__(observation_space, features_dim)
        # observation_space.shape = (C, H, W)
        n_input_channels = observation_space.shape[0]
        self.cnn = ImpalaCNN(input_channels=n_input_channels)

        # Compute the size of the output of the CNN by passing a dummy tensor
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            cnn_output = self.cnn(sample_input)
        self._features_dim = cnn_output.shape[1]

    def forward(self, observations):
        return self.cnn(observations)


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

class CustomMlpPolicy(ActorCriticPolicy):
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
    elif args.nn == 'CustomMlpPolicy':
        size = args.nnsize
        nn_type = CustomMlpPolicy
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[size, size], dropout_prob=0.3)
    elif args.nn == 'CustomCnnPolicy':
        #size = args.nnsize
        nn_type = 'CnnPolicy'
        policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=128),)
    elif args.nn == 'ImpalaCnnPolicy':
        #size = args.nnsize
        nn_type = 'CnnPolicy'
        policy_kwargs = dict(features_extractor_class=CustomImpalaFeatureExtractor,)
    elif args.nn == 'CombinedPolicy':
        #size = args.nnsize
        nn_type = CustomPolicy
        policy_kwargs = {}

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
