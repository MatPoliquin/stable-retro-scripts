import os
from stable_baselines3 import PPO, A2C
import torch
import torch as th
import torch.nn as nn
from torchsummary import summary
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
import timm
import json

# ==========================================================================================
class AttentionMLP(BaseFeaturesExtractor):
    """
    Custom MLP with self-attention for hockey.
    Input: Structured data (e.g., player positions, velocities).
    Output: Feature vector for RL policy/value heads.
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.num_features = observation_space.shape[0]

        # Self-attention layer
        self.query = nn.Linear(self.num_features, self.num_features)
        self.key = nn.Linear(self.num_features, self.num_features)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Self-attention
        q = self.query(x)  # [batch_size, num_features]
        k = self.key(x)    # [batch_size, num_features]

        # Compute attention scores
        # q.unsqueeze(1): [batch_size, 1, num_features]
        # k.unsqueeze(2): [batch_size, num_features, 1]
        # bmm result: [batch_size, 1, 1]
        attention_scores = torch.bmm(q.unsqueeze(1), k.unsqueeze(2)) / (self.num_features ** 0.5)

        # Apply softmax - need to specify dimension
        weights = torch.softmax(attention_scores, dim=-1)  # Apply along last dimension

        # Apply attention weights
        attended_x = weights.squeeze(-1) * x  # [batch_size, num_features]

        # MLP
        return self.mlp(attended_x)

class AttentionMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=AttentionMLP,
            features_extractor_kwargs={"features_dim": 128},
        )
# ==========================================================================================
class DartFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dart model that uses decision trees for feature extraction.
    """
    def __init__(self, observation_space, features_dim=64):
        super(DartFeatureExtractor, self).__init__(observation_space, features_dim)

        # Determine input dimension based on observation space
        if isinstance(observation_space, gym.spaces.Dict):
            self.is_dict = True
            self.flatten = nn.Flatten()
            with th.no_grad():
                sample = observation_space.sample()
                sample_tensor = {k: th.as_tensor(v[None]).float() for k, v in sample.items()}
                flattened_dim = sum(v.numel() for v in sample_tensor.values())
        else:
            self.is_dict = False
            with th.no_grad():
                sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
                flattened_dim = sample_tensor.numel()

        # Tree-like layers with residual connections
        self.tree_layer1 = nn.Sequential(
            nn.Linear(flattened_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )

        self.tree_layer2 = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )

        self._features_dim = features_dim

    def forward(self, observations):
        if self.is_dict:
            flattened = []
            for key, value in observations.items():
                flattened.append(self.flatten(value))
            x = th.cat(flattened, dim=1)
        else:
            x = observations.flatten(start_dim=1)

        out1 = self.tree_layer1(x)
        out2 = self.tree_layer2(out1)
        return out1 + out2  # Residual connection

# ==========================================================================================
class DartPolicy(ActorCriticPolicy):
    """
    Policy network for Dart model with tree-like feature extraction.
    """
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, *args, **kwargs):
        # Remove features_extractor_kwargs from kwargs if present
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        features_extractor_kwargs['features_dim'] = features_extractor_kwargs.get('features_dim', 64)

        super(DartPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch or dict(pi=[64, 64], vf=[64, 64]),
            features_extractor_class=DartFeatureExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs
        )

        # Get the actual latent dimension from mlp_extractor
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf

        # Additional tree-like processing in policy head
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim_pi, latent_dim_pi),
            nn.ReLU(),
            nn.Linear(latent_dim_pi, latent_dim_pi),
            nn.ReLU()
        )

        # Value head should output a single value
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim_vf, latent_dim_vf),
            nn.ReLU(),
            nn.Linear(latent_dim_vf, 1),  # Output single value
            nn.ReLU()
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Additional tree-like processing
        latent_pi = self.policy_net(latent_pi)

        # Get value estimate (should be scalar per observation)
        values = self.value_net(latent_vf).squeeze(-1)  # Remove extra dimension

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def predict_values(self, obs):
        """
        Get the value estimates for observations
        """
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf).squeeze(-1)

# ==========================================================================================
class ViTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(ViTFeatureExtractor, self).__init__(observation_space, features_dim)

        # Expecting image observation only (C, H, W)
        channels, height, width = observation_space.shape

        # Load pretrained ViT base model with patch size 16 (adjust as you see fit)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Adjust input conv if there is channel mismatch
        if channels != 3:
            self.vit.patch_embed.proj = nn.Conv2d(channels, self.vit.embed_dim, kernel_size=16, stride=16)

        # Remove classifier head, keep feature extraction only
        self.vit.head = nn.Identity()

        # Project ViT output dimension to desired features_dim
        self.linear = nn.Linear(self.vit.embed_dim, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Resize input to 224x224 as ViT expects fixed input size
        x = nn.functional.interpolate(observations, size=(224, 224), mode='bilinear', align_corners=False)

        features = self.vit(x)
        return self.linear(features)


class ViTPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(ViTPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=ViTFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
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

def load_hyperparameters(json_file):
    with open(json_file, 'r') as f:
        hyperparams = json.load(f)

    print(hyperparams)

    return hyperparams

def print_model_summary(env, player_model, model):
    # Print model summary
    if player_model == '':  # Only for newly created models
        print("\nModel Architecture Summary:")

        # Get the policy network
        policy = model.policy

        # Print basic info
        print(f"\nPolicy Network Type: {policy.__class__.__name__}")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")

        # Print feature extractor info if available
        if hasattr(policy, 'features_extractor'):
            print("\nFeature Extractor Architecture:")
            fe = policy.features_extractor
            print(f"Type: {fe.__class__.__name__}")

            # Print layers for common extractor types
            if isinstance(fe, (CustomCNN, CustomImpalaFeatureExtractor, ViTFeatureExtractor)):
                for name, layer in fe.named_children():
                    print(f"  {name}: {layer.__class__.__name__}")
                    if hasattr(layer, 'weight'):
                        print(f"    Weight shape: {layer.weight.shape}")

            # Special handling for CNN-based extractors
            if hasattr(fe, 'cnn'):
                print("\nCNN Layers:")
                for name, layer in fe.cnn.named_children():
                    print(f"  {name}: {layer.__class__.__name__}")
                    if isinstance(layer, nn.Conv2d):
                        print(f"    in_channels: {layer.in_channels}")
                        print(f"    out_channels: {layer.out_channels}")
                        print(f"    kernel_size: {layer.kernel_size}")

        # Print MLP extractor info if available
        if hasattr(policy, 'mlp_extractor'):
            print("\nMLP Extractor Architecture:")
            mlp = policy.mlp_extractor
            print(f"Type: {mlp.__class__.__name__}")

            if hasattr(mlp, 'shared_net'):
                print("\nShared Layers:")
                for i, layer in enumerate(mlp.shared_net):
                    print(f"  Layer {i}: {layer.__class__.__name__}")
                    if hasattr(layer, 'weight'):
                        print(f"    Weight shape: {layer.weight.shape}")

            if hasattr(mlp, 'policy_net'):
                print("\nPolicy Head:")
                for i, layer in enumerate(mlp.policy_net):
                    print(f"  Layer {i}: {layer.__class__.__name__}")

            if hasattr(mlp, 'value_net'):
                print("\nValue Head:")
                for i, layer in enumerate(mlp.value_net):
                    print(f"  Layer {i}: {layer.__class__.__name__}")

        # Print action distribution info
        if hasattr(policy, 'action_dist'):
            print("\nAction Distribution:")
            print(f"Type: {policy.action_dist.__class__.__name__}")

        # Print total parameters
        total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"\nTotal Trainable Parameters: {total_params:,}")

def init_model(output_path, player_model, player_alg, args, env, logger):
    policy_kwargs = None
    nn_type = args.nn
    size = args.nnsize

    # Load hyperparameters from JSON
    if args.hyperparams:
        if os.path.isfile(args.hyperparams):
            hyperparams = load_hyperparameters(args.hyperparams)
        else:
            raise FileNotFoundError(f"Hyperparameters file not found: {args.hyperparams}")
    else:
        hyperparams = {}

    if args.nn == 'MlpPolicy':
        nn_type = 'MlpPolicy'
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size]))
        )
    elif args.nn == 'CustomMlpPolicy':
        nn_type = CustomMlpPolicy
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=hyperparams.get("net_arch", [size, size]),
            dropout_prob=hyperparams.get("dropout_prob", 0.3)
        )
    elif args.nn == 'CustomCnnPolicy':
        nn_type = 'CnnPolicy'
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=hyperparams.get('features_dim', 128))
        )
    elif args.nn == 'ImpalaCnnPolicy':
        nn_type = 'CnnPolicy'
        policy_kwargs = dict(features_extractor_class=CustomImpalaFeatureExtractor)
    elif args.nn == 'CombinedPolicy':
        nn_type = CustomPolicy
        policy_kwargs = {}
    elif args.nn == 'ViTPolicy':
        nn_type = ViTPolicy
        policy_kwargs = {}
    elif args.nn == 'DartPolicy':
        nn_type = DartPolicy
        policy_kwargs = dict(
            net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size])),
            features_extractor_kwargs=dict(features_dim=hyperparams.get('features_dim', 128))
        )
    elif args.nn == 'AttentionMLPPolicy':
        nn_type = AttentionMLPPolicy
        policy_kwargs = {}

    if player_alg == 'ppo2':
        if player_model == '':
            batch_size = hyperparams.get('batch_size', 256) * args.num_env
            print("batch_size:%d" % batch_size)
            model = PPO(
                policy=nn_type,
                env=env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                n_steps=hyperparams.get('n_steps', 2048),
                n_epochs=hyperparams.get('n_epochs', 4),
                batch_size=batch_size,
                learning_rate=hyperparams.get('learning_rate', 2.5e-4),
                clip_range=hyperparams.get('clip_range', 0.2),
                vf_coef=hyperparams.get('vf_coef', 0.5),
                ent_coef=hyperparams.get('ent_coef', 0.01),
                max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
                clip_range_vf=hyperparams.get('clip_range_vf', None)
            )
        else:
            model = PPO.load(os.path.expanduser(player_model), env=env)
    elif player_alg == 'a2c':
        if player_model == '':
            model = A2C(policy=nn_type, env=env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            model = A2C.load(os.path.expanduser(player_model), env=env, verbose=1, tensorboard_log=output_path)


    print_model_summary(env, player_model, model)

    model.set_logger(logger)
    return model
