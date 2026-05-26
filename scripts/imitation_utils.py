import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import torch as th
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor

import game_wrappers_mgr as games
from env_utils import isMLP, make_retro, resolve_backend_action_type
from env_wrappers import RewardClipper, StochasticFrameSkip, WarpFrameDict
from game_wrappers.nhl94.nhl94_const import GameConsts
from utils import resolve_clip_reward, resolve_sticky_action_settings


RARE_BUTTON_INDICES = (
    GameConsts.INPUT_A,
    GameConsts.INPUT_B,
    GameConsts.INPUT_C,
    GameConsts.INPUT_MODE,
)


NHL94_DEFAULT_STATES = {
    "NHL941on1-Genesis-v0": "PenguinsVsSenators.start",
    "NHL942on2-Genesis-v0": "PenguinsVsWhalers.start",
    "NHL94-Genesis-v0": "PenguinsVsSenators.start",
}


def resolve_default_state(args):
    if getattr(args, "state", None) is not None:
        return args.state

    default_state = NHL94_DEFAULT_STATES.get(getattr(args, "env", ""))
    if default_state is not None:
        args.state = default_state
    return getattr(args, "state", None)


def build_single_nhl94_env(
    args,
    hyperparams,
    *,
    num_players=1,
    output_path=None,
    use_sticky_action=False,
    use_frame_skip=True,
):
    games.wrappers.init(args)
    resolve_default_state(args)
    backend_action_type = resolve_backend_action_type(args, num_players)
    env = make_retro(
        game=args.env,
        action_type=backend_action_type,
        state=args.state,
        num_players=num_players,
    )

    if isMLP(args.nn):
        env = games.wrappers.obs_env(env, args, num_players, args.rf)

    if output_path:
        env = Monitor(env, output_path, allow_early_resets=True)

    if use_frame_skip:
        frame_skip = max(1, int(hyperparams.get("frame_skip", 4)))
        sticky_enabled, sticky_prob = resolve_sticky_action_settings(
            use_sticky_action,
            hyperparams,
        )
        env = StochasticFrameSkip(env, n=frame_skip, stickprob=sticky_prob if sticky_enabled else -1)

    if not isMLP(args.nn):
        env = WarpFrame(env)
    elif args.nn == "CombinedPolicy":
        env = WarpFrameDict(env)

    if resolve_clip_reward(args, hyperparams):
        env = RewardClipper(env, low=-1.0, high=1.0)

    return env


def find_wrapper_with_attr(env, attr_name):
    current = env
    while current is not None:
        if hasattr(current, attr_name):
            return current
        current = getattr(current, "env", None)
    raise AttributeError(f"Could not find wrapper attribute: {attr_name}")


def get_game_state(env):
    return find_wrapper_with_attr(env, "game_state").game_state


def normalize_reset_result(reset_result):
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


def normalize_step_result(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = step_result
    return obs, reward, bool(done), info


def ensure_box_observation(obs, nn_name):
    if isinstance(obs, dict):
        raise NotImplementedError(
            f"BC dataset collection currently expects Box observations; got Dict for {nn_name}."
        )
    return np.asarray(obs, dtype=np.float32)


def save_demo_shard(path, arrays, metadata):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    manifest_path = f"{path}.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return path


def discover_npz_files(paths):
    files = []
    for path in paths:
        expanded = os.path.expanduser(path)
        if os.path.isdir(expanded):
            files.extend(sorted(glob.glob(os.path.join(expanded, "*.npz"))))
        else:
            matches = sorted(glob.glob(expanded))
            files.extend(matches if matches else [expanded])
    files = [path for path in files if path.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No .npz dataset shards found in: {paths}")
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Dataset shard not found: {missing[0]}")
    return files


def load_demo_arrays(paths):
    observations = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []

    next_episode_offset = 0
    for path in discover_npz_files(paths):
        with np.load(path, allow_pickle=False) as data:
            observations.append(np.asarray(data["observations"], dtype=np.float32))
            actions.append(np.asarray(data["actions"], dtype=np.float32))
            if "rewards" in data:
                rewards.append(np.asarray(data["rewards"], dtype=np.float32))
            if "dones" in data:
                dones.append(np.asarray(data["dones"], dtype=np.bool_))
            if "episode_ids" in data:
                shard_episode_ids = np.asarray(data["episode_ids"], dtype=np.int64) + next_episode_offset
                episode_ids.append(shard_episode_ids)
                next_episode_offset = int(shard_episode_ids.max(initial=-1)) + 1

    payload = {
        "observations": np.concatenate(observations, axis=0),
        "actions": np.concatenate(actions, axis=0),
    }
    if rewards:
        payload["rewards"] = np.concatenate(rewards, axis=0)
    if dones:
        payload["dones"] = np.concatenate(dones, axis=0)
    if episode_ids:
        payload["episode_ids"] = np.concatenate(episode_ids, axis=0)
    else:
        payload["episode_ids"] = np.zeros(payload["actions"].shape[0], dtype=np.int64)
    return payload


def split_by_episode(episode_ids, validation_fraction, seed):
    unique_ids = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    val_count = max(1, int(round(len(unique_ids) * validation_fraction))) if len(unique_ids) > 1 else 0
    val_ids = set(unique_ids[:val_count].tolist())
    val_mask = np.asarray([episode_id in val_ids for episode_id in episode_ids], dtype=bool)
    train_mask = ~val_mask
    if not train_mask.any():
        train_mask[:] = True
        val_mask[:] = False
    return train_mask, val_mask


def compute_sample_weights(actions, rare_weight):
    weights = np.ones(actions.shape[0], dtype=np.float32)
    rare_mask = actions[:, RARE_BUTTON_INDICES].sum(axis=1) > 0
    weights[rare_mask] = float(rare_weight)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def policy_action_logits(policy, observations):
    features = policy.extract_features(observations)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_pi)


def button_metrics(logits, actions):
    probabilities = th.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    targets = actions.float()

    matches = predictions.eq(targets)
    exact_match = matches.all(dim=1).float().mean().item()
    hamming_accuracy = matches.float().mean().item()

    intersection = (predictions * targets).sum(dim=1)
    union = ((predictions + targets) > 0).float().sum(dim=1)
    jaccard = th.where(union > 0, intersection / union.clamp_min(1.0), th.ones_like(union))

    tp = (predictions * targets).sum(dim=0)
    fp = (predictions * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - predictions) * targets).sum(dim=0)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)

    return {
        "exact_match": exact_match,
        "hamming_accuracy": hamming_accuracy,
        "jaccard": jaccard.mean().item(),
        "per_button_precision": precision.detach().cpu().tolist(),
        "per_button_recall": recall.detach().cpu().tolist(),
        "per_button_f1": f1.detach().cpu().tolist(),
        "rare_button_f1": f1[list(RARE_BUTTON_INDICES)].mean().item(),
    }


def namespace_with_defaults(args, **updates):
    payload = vars(args).copy()
    payload.update(updates)
    return SimpleNamespace(**payload)