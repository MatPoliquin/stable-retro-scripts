#!/usr/bin/env python3
"""Behavior-clone NHL94 ClassicAI demonstrations into an SB3 policy."""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from torch.utils.data import DataLoader, TensorDataset

from imitation_utils import (
    build_single_nhl94_env,
    button_metrics,
    compute_sample_weights,
    load_demo_arrays,
    policy_action_logits,
    split_by_episode,
)
from game_wrappers.nhl94.nhl94_const import GameConsts
from models_utils import init_model
from utils import load_hyperparams


GAMESTATE_BUTTON_TO_ACTION_INDEX = (
    GameConsts.INPUT_UP,
    GameConsts.INPUT_DOWN,
    GameConsts.INPUT_LEFT,
    GameConsts.INPUT_RIGHT,
    GameConsts.INPUT_B,
    GameConsts.INPUT_C,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Behavior-clone ClassicAI demonstrations")
    parser.add_argument("--datasets", nargs="+", required=True, help="Demo .npz files, directories, or globs")
    parser.add_argument("--output_model", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="~/OUTPUT/classic_ai_bc")
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--env", type=str, default="NHL94-Genesis-v0")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--rf", type=str, default="PostPlay")
    parser.add_argument("--nn", type=str, default="ResidualMlpPolicy")
    parser.add_argument("--alg", type=str, default="ppo2")
    parser.add_argument("--nnsize", type=int, default=256)
    parser.add_argument("--num_players", type=int, default=1)
    parser.add_argument("--hyperparams", type=str, default="../hyperparams/nhl94_residual_mlp.json")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--action_type", type=str, default="FILTERED", choices=["FILTERED", "DISCRETE", "MULTI_DISCRETE"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--rare_weight", type=float, default=4.0)
    parser.add_argument("--positive_weight_max", type=float, default=32.0)
    parser.add_argument("--release_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip_reward", default=None, action="store_true")
    parser.add_argument("--no_clip_reward", dest="clip_reward", action="store_false")
    parser.add_argument("--no_frame_skip", default=False, action="store_true")
    return parser


def parse_cmdline(argv):
    return build_parser().parse_args(argv)


def _make_output_model_path(args):
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.output_model:
        return os.path.expanduser(args.output_model)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(output_dir, f"classic_ai_bc_{args.env}_{args.rf}_{args.nn}_{timestamp}.zip")


def _make_model(args, hyperparams, env, logger):
    if args.load_model:
        model = PPO.load(os.path.expanduser(args.load_model), env=env, device=args.device)
        model.set_logger(logger)
        return model
    return init_model(None, "", args.alg, args, env, logger, hyperparams)


def _button_state_offset(num_players):
    return 15 * int(num_players) + 15


def _compute_element_weights(observations, actions, release_weight, num_players):
    weights = np.ones_like(actions, dtype=np.float32)
    if release_weight <= 1.0:
        return weights

    button_offset = _button_state_offset(num_players)
    if observations.ndim != 2 or observations.shape[1] < button_offset + len(GAMESTATE_BUTTON_TO_ACTION_INDEX):
        return weights

    previous_buttons = observations[:, button_offset : button_offset + len(GAMESTATE_BUTTON_TO_ACTION_INDEX)]
    for button_state_index, action_index in enumerate(GAMESTATE_BUTTON_TO_ACTION_INDEX):
        release_mask = (previous_buttons[:, button_state_index] > 0.5) & (actions[:, action_index] < 0.5)
        weights[release_mask, action_index] = float(release_weight)
    return weights


def _make_loader(observations, actions, weights, element_weights, batch_size, shuffle):
    obs_tensor = th.as_tensor(observations, dtype=th.float32)
    action_tensor = th.as_tensor(actions, dtype=th.float32)
    weight_tensor = th.as_tensor(weights, dtype=th.float32)
    element_weight_tensor = th.as_tensor(element_weights, dtype=th.float32)
    dataset = TensorDataset(obs_tensor, action_tensor, weight_tensor, element_weight_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _compute_positive_weights(actions, max_weight):
    positive_counts = actions.sum(axis=0).astype(np.float32)
    negative_counts = actions.shape[0] - positive_counts
    weights = np.ones_like(positive_counts, dtype=np.float32)
    positive_mask = positive_counts > 0
    weights[positive_mask] = negative_counts[positive_mask] / positive_counts[positive_mask]
    return np.clip(weights, 1.0, float(max_weight))


def _evaluate(model, loader, criterion, device):
    model.policy.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_actions = []
    with th.no_grad():
        for obs_batch, action_batch, weight_batch, element_weight_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            weight_batch = weight_batch.to(device)
            element_weight_batch = element_weight_batch.to(device)
            logits = policy_action_logits(model.policy, obs_batch)
            per_button_loss = criterion(logits, action_batch) * element_weight_batch
            per_sample_loss = per_button_loss.mean(dim=1)
            loss = (per_sample_loss * weight_batch).mean()
            batch_size = obs_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_logits.append(logits.detach().cpu())
            all_actions.append(action_batch.detach().cpu())

    metrics = button_metrics(th.cat(all_logits, dim=0), th.cat(all_actions, dim=0))
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def train_bc(args, hyperparams):
    if args.action_type != "FILTERED":
        raise NotImplementedError("BC pretraining currently expects --action_type=FILTERED.")

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    arrays = load_demo_arrays(args.datasets)
    observations = arrays["observations"].astype(np.float32)
    actions = arrays["actions"].astype(np.float32)
    episode_ids = arrays["episode_ids"]

    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError(f"Expected actions with shape (N, 12), got {actions.shape}")

    weights = compute_sample_weights(actions, args.rare_weight)
    element_weights = _compute_element_weights(
        observations,
        actions,
        args.release_weight,
        args.num_players,
    )
    train_mask, val_mask = split_by_episode(episode_ids, args.validation_fraction, args.seed)

    train_loader = _make_loader(
        observations[train_mask],
        actions[train_mask],
        weights[train_mask],
        element_weights[train_mask],
        args.batch_size,
        shuffle=True,
    )
    val_loader = None
    if val_mask.any():
        val_loader = _make_loader(
            observations[val_mask],
            actions[val_mask],
            weights[val_mask],
            element_weights[val_mask],
            args.batch_size,
            shuffle=False,
        )

    output_model = _make_output_model_path(args)
    run_dir = os.path.dirname(os.path.abspath(output_model))
    logger = configure(run_dir, ["stdout"])
    env = build_single_nhl94_env(
        args,
        hyperparams,
        num_players=args.num_players,
        use_sticky_action=False,
        use_frame_skip=not args.no_frame_skip,
    )

    try:
        model = _make_model(args, hyperparams, env, logger)
        device = model.policy.device
        optimizer = th.optim.Adam(model.policy.parameters(), lr=args.learning_rate)
        pos_weight = th.as_tensor(
            _compute_positive_weights(actions[train_mask], args.positive_weight_max),
            dtype=th.float32,
            device=device,
        )
        criterion = th.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_model = output_model
        history = []

        for epoch in range(1, args.epochs + 1):
            model.policy.train()
            total_loss = 0.0
            total_samples = 0
            for obs_batch, action_batch, weight_batch, element_weight_batch in train_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)
                weight_batch = weight_batch.to(device)
                element_weight_batch = element_weight_batch.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = policy_action_logits(model.policy, obs_batch)
                per_button_loss = criterion(logits, action_batch) * element_weight_batch
                per_sample_loss = per_button_loss.mean(dim=1)
                loss = (per_sample_loss * weight_batch).mean()
                loss.backward()
                optimizer.step()

                batch_size = obs_batch.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

            train_metrics = _evaluate(model, train_loader, criterion, device)
            train_metrics["loss"] = total_loss / max(total_samples, 1)
            val_metrics = _evaluate(model, val_loader, criterion, device) if val_loader is not None else None
            current_val_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]

            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))

            if current_val_loss <= best_val_loss:
                best_val_loss = current_val_loss
                model.save(output_model)
                best_model = output_model

        metrics_path = f"{output_model}.metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_model": best_model,
                    "best_val_loss": best_val_loss,
                    "num_samples": int(actions.shape[0]),
                    "num_train_samples": int(train_mask.sum()),
                    "num_validation_samples": int(val_mask.sum()),
                    "positive_weights": pos_weight.detach().cpu().tolist(),
                    "release_weight": float(args.release_weight),
                    "history": history,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        print(json.dumps({"saved": best_model, "metrics": metrics_path}, indent=2, sort_keys=True))
        return best_model
    finally:
        env.close()


def main(argv):
    args = parse_cmdline(argv[1:])
    hyperparams = load_hyperparams(
        args.hyperparams,
        required=True,
        base_dir=os.path.dirname(__file__),
    )
    train_bc(args, hyperparams)


if __name__ == "__main__":
    main(sys.argv)