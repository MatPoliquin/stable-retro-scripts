import os
from stable_baselines3 import PPO, A2C
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchsummary import summary
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
import timm
import json

class CNNTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, nhead=8, num_layers=3):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output dim
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # Transformer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=cnn_out_dim, nhead=nhead, dim_feedforward=256, dropout=0.1
        )
        self.transformer = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(cnn_out_dim, features_dim)

    def forward(self, observations):
        # CNN extracts spatial features
        cnn_features = self.cnn(observations)  # Shape: (batch_size, cnn_out_dim)

        # Transformer expects (seq_len, batch_size, dim)
        # We treat each frame as a "sequence of 1" for simplicity
        cnn_features = cnn_features.unsqueeze(0)  # Shape: (1, batch_size, cnn_out_dim)

        # Transformer processes temporal features
        transformer_out = self.transformer(cnn_features)
        transformer_out = transformer_out.squeeze(0)  # Back to (batch_size, cnn_out_dim)

        # Final projection
        return self.fc(transformer_out)

# ==========================================================================================
class EntityAttentionMLP(BaseFeaturesExtractor):
    """
    Enhanced attention MLP with both temporal and intra-frame (entity) attention.
    Processes structured observations (like Pong's [paddle1, paddle2, ball]).
    """
    def __init__(self, observation_space, features_dim=128, num_frames=4):
        super().__init__(observation_space, features_dim)
        self.num_frames = num_frames
        self.input_dim = observation_space.shape[0] // num_frames  # Features per frame

        # Entity projections (paddle1, paddle2, ball)
        self.entity_proj = nn.Sequential(
            nn.Linear(2, 32),  # Each entity represented by 2 values
            nn.ReLU()
        )

        # Temporal attention across frames
        self.temp_attention = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Entity attention within frames
        self.entity_attention = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Final feature processor
        self.feature_net = nn.Sequential(
            nn.Linear(32, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Reshape: (batch, num_frames*input_dim) -> (batch, num_frames, input_dim)
        x = x.view(-1, self.num_frames, self.input_dim)

        # Split into entities per frame: [p1_y, p2_y, ball_x, ball_y, ball_vx, ball_vy]
        # -> 3 entities: paddle1, paddle2, ball (position + velocity)
        entities = torch.stack([
            x[..., [0, 1]],  # Paddle1 (y + dummy)
            x[..., [1, 0]],  # Paddle2 (y + dummy)
            x[..., [2, 3]],  # Ball position
            x[..., [4, 5]],  # Ball velocity
        ], dim=2)  # Shape: (batch, num_frames, 4 entities, 2)

        # Project entities
        entity_embeds = self.entity_proj(entities)  # (batch, num_frames, 4, 32)

        # Intra-frame attention (ball vs paddles)
        entity_weights = F.softmax(self.entity_attention(entity_embeds), dim=2)
        frame_embeds = torch.sum(entity_weights * entity_embeds, dim=2)  # (batch, num_frames, 32)

        # Temporal attention across frames
        temp_weights = F.softmax(self.temp_attention(frame_embeds), dim=1)
        context = torch.sum(temp_weights * frame_embeds, dim=1)  # (batch, 32)

        return self.feature_net(context)

class EntityAttentionPolicy(ActorCriticPolicy):
    def __init__(self, *args, num_frames=4, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=EntityAttentionMLP,
            features_extractor_kwargs={
                "features_dim": 128,
                "num_frames": num_frames
            }
        )
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

