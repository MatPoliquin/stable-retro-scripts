import os
import gymnasium as gym
import torch as th

# Avoid noisy NNPACK init warnings on some CPUs (common on some AMD systems)
# and force PyTorch to use other backends.
try:
    import torch.backends.nnpack as nnpack

    nnpack.set_flags(False)
except Exception:
    pass

from stable_baselines3 import PPO, A2C
from torchsummary import summary
from models import CustomMlpPolicy, CustomPolicy, ViTPolicy, DartPolicy, AttentionMLPPolicy,\
    EntityAttentionPolicy, CustomCNN, CustomImpalaFeatureExtractor, CNNTransformer, HockeyMultiHeadPolicy
from es import EvolutionStrategies

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

def print_model_summary(args, env, player_model, model):

    obs_space = getattr(model, "observation_space", None) or env.observation_space
    print(obs_space)

    if args.alg == 'es':
        return

    # Handle policies
    if args.nn in ('MlpPolicy', 'AttentionMLPPolicy', 'EntityAttentionPolicy', 'CustomMlpPolicy'):
        obs_shape = obs_space.shape
        pytorch_obs_shape = (obs_shape[0],)
    elif args.nn in ('CnnPolicy', 'ImpalaCnnPolicy'):
        # SB3 model.observation_space is already channels-first and includes frame stack
        obs_shape = obs_space.shape  # (C, H, W)
        pytorch_obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2])
    else:
        # Other types are not supported for summary for now
        return

    summary(model.policy, pytorch_obs_shape)

def init_model(output_path, player_model, player_alg, args, env, logger, hyperparams):
    policy_kwargs = None
    nn_type = args.nn
    size = args.nnsize

    if hyperparams is None:
        raise ValueError("hyperparams must be provided; load them via utils.load_hyperparams")

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
        attention_kwargs = hyperparams.get('attention_mlp', {}) if hyperparams else {}
        if not isinstance(attention_kwargs, dict):
            raise TypeError("hyperparams['attention_mlp'] must be a dict when provided")
        features_extractor_kwargs = dict(attention_kwargs)
        features_extractor_kwargs.setdefault('features_dim', hyperparams.get('features_dim', 128))

        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size])),
            features_extractor_kwargs=features_extractor_kwargs
        )
    elif args.nn == 'EntityAttentionPolicy':
        nn_type = EntityAttentionPolicy
        policy_kwargs = {}
    elif args.nn == 'CnnTransformerPolicy':
        nn_type = 'CnnPolicy'
        policy_kwargs = dict(
            features_extractor_class=CNNTransformer,
            features_extractor_kwargs=dict(features_dim=512, nhead=8, num_layers=3),
        )
    elif args.nn == 'HockeyMultiHeadPolicy':
        nn_type = HockeyMultiHeadPolicy
        policy_kwargs = dict(
            features_extractor_kwargs=dict()
        )

    if player_alg == 'ppo2':
        if player_model == '':
            batch_size = hyperparams.get('batch_size', 256)
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
                clip_range_vf=hyperparams.get('clip_range_vf', None),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                normalize_advantage=hyperparams.get('normalize_advantage', True),
                target_kl=hyperparams.get('target_kl', None)
            )
        else:
            model = PPO.load(os.path.expanduser(player_model), env=env)

        model.set_logger(logger)

    elif player_alg == 'es':
        es = EvolutionStrategies(env, args, 1, None)
        return es

    print_model_summary(args, env.unwrapped, player_model, model)


    return model
