#!/usr/bin/env python3
"""Collect NHL94 ClassicAI demonstrations for imitation learning."""

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

from classic_ai import ClassicAIModel
from game_wrappers.nhl94.nhl94_intents import HOCKEY_INTENT_DPAD_ACTIONS
from imitation_utils import (
    build_single_nhl94_env,
    ensure_box_observation,
    get_game_state,
    normalize_reset_result,
    normalize_step_result,
    save_demo_shard,
)
from utils import load_hyperparams


def build_parser():
    parser = argparse.ArgumentParser(description="Collect NHL94 ClassicAI demonstration shards")
    parser.add_argument("--env", type=str, default="NHL94-Genesis-v0")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--rf", type=str, default="PostPlay")
    parser.add_argument("--nn", type=str, default="ResidualMlpPolicy")
    parser.add_argument("--alg", type=str, default="ppo2")
    parser.add_argument("--nnsize", type=int, default=256)
    parser.add_argument("--num_players", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=4500)
    parser.add_argument("--output", type=str, default="~/OUTPUT/classic_ai_demos")
    parser.add_argument("--shard_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hyperparams", type=str, default="../hyperparams/nhl94_residual_mlp.json")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--action_type", type=str, default="HOCKEY_INTENT_DPAD", choices=["FILTERED", "DISCRETE", "MULTI_DISCRETE", "HOCKEY_INTENT_DPAD"])
    parser.add_argument("--opponent", type=str, default="game", choices=["game", "noop"])
    parser.add_argument("--no_frame_skip", default=False, action="store_true")
    parser.add_argument("--clip_reward", default=None, action="store_true")
    parser.add_argument("--no_clip_reward", dest="clip_reward", action="store_false")
    return parser


def parse_cmdline(argv):
    return build_parser().parse_args(argv)


def _validate_args(args):
    args.action_type = args.action_type.upper()
    if args.action_type not in ("FILTERED", "HOCKEY_INTENT_DPAD"):
        raise NotImplementedError("ClassicAI cloning currently supports FILTERED and HOCKEY_INTENT_DPAD.")
    if args.action_type == "HOCKEY_INTENT_DPAD" and args.opponent == "noop":
        raise ValueError("HOCKEY_INTENT_DPAD currently supports --opponent=game only.")
    if args.action_type == "HOCKEY_INTENT_DPAD" and args.num_players != 1:
        raise ValueError("HOCKEY_INTENT_DPAD currently expects --num_players=1.")
    if args.opponent == "noop" and args.num_players != 2:
        args.num_players = 2
    if args.opponent == "game" and args.num_players != 1:
        raise ValueError("--opponent=game expects --num_players=1 so the built-in game AI controls the opponent.")


def _make_output_path(args):
    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)
    if args.shard_name:
        shard_name = args.shard_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shard_name = f"classic_ai_{args.env}_{args.rf}_{args.nn}_{timestamp}.npz"
    if not shard_name.endswith(".npz"):
        shard_name += ".npz"
    return os.path.join(output_dir, shard_name)


def _split_episode_ids(num_episodes, num_workers):
    worker_count = max(1, min(int(num_workers), int(num_episodes)))
    chunks = [[] for _ in range(worker_count)]
    for index, episode_id in enumerate(range(int(num_episodes))):
        chunks[index % worker_count].append(episode_id)
    return [chunk for chunk in chunks if chunk]


def _collect_episode_batch(args_payload, hyperparams, episode_ids):
    args = argparse.Namespace(**args_payload)
    np.random.seed(args.seed + min(episode_ids))
    env = build_single_nhl94_env(
        args,
        hyperparams,
        num_players=args.num_players,
        use_sticky_action=False,
        use_frame_skip=not args.no_frame_skip,
    )
    expert = ClassicAIModel(args=args, env=env)

    observations = []
    actions = []
    rewards = []
    dones = []
    sample_episode_ids = []
    steps = []
    decisions = []
    targets = []

    try:
        for episode_id in episode_ids:
            reset_result = env.reset(seed=args.seed + episode_id)
            obs, _ = normalize_reset_result(reset_result)

            for step in range(args.max_steps):
                obs_array = ensure_box_observation(obs, args.nn)
                game_state = get_game_state(env)
                action = np.asarray(expert.predict_game_state(game_state)[0], dtype=np.int8)

                observations.append(obs_array.copy())
                actions.append(action.copy())
                sample_episode_ids.append(episode_id)
                steps.append(step)
                decisions.append(str(getattr(expert, "_last_decision", "")))
                targets.append(np.asarray(getattr(expert, "_last_target", (0, 0)), dtype=np.float32))

                obs, reward, done, _ = normalize_step_result(env.step(action))
                rewards.append(float(reward))
                dones.append(bool(done))

                if done:
                    break
    finally:
        env.close()

    if not observations:
        raise RuntimeError("No demonstrations were collected.")

    return {
        "observations": np.stack(observations).astype(np.float32),
        "actions": np.stack(actions).astype(np.int8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "episode_ids": np.asarray(sample_episode_ids, dtype=np.int64),
        "steps": np.asarray(steps, dtype=np.int64),
        "decisions": np.asarray(decisions, dtype="U64"),
        "targets": np.stack(targets).astype(np.float32),
    }


def _concat_arrays(shards):
    keys = shards[0].keys()
    return {key: np.concatenate([shard[key] for shard in shards], axis=0) for key in keys}


def collect_demos(args, hyperparams):
    _validate_args(args)
    worker_count = max(1, int(getattr(args, "num_workers", 1)))

    episode_chunks = _split_episode_ids(args.num_episodes, worker_count)
    if len(episode_chunks) == 1:
        arrays = _collect_episode_batch(vars(args), hyperparams, episode_chunks[0])
    else:
        print(f"Collecting {args.num_episodes} episodes with {len(episode_chunks)} workers", flush=True)
        arrays_by_chunk = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(episode_chunks), mp_context=context) as executor:
            futures = [
                executor.submit(_collect_episode_batch, vars(args), hyperparams, chunk)
                for chunk in episode_chunks
            ]
            for future in as_completed(futures):
                arrays_by_chunk.append(future.result())
        arrays_by_chunk.sort(key=lambda shard: int(shard["episode_ids"].min()))
        arrays = _concat_arrays(arrays_by_chunk)

    arrays = {
        key: arrays[key]
        for key in (
            "observations",
            "actions",
            "rewards",
            "dones",
            "episode_ids",
            "steps",
            "decisions",
            "targets",
        )
    }
    action_counts = arrays["actions"].sum(axis=0).astype(int).tolist()
    metadata = {
        "env": args.env,
        "state": args.state,
        "rf": args.rf,
        "nn": args.nn,
        "action_type": args.action_type,
        "opponent": args.opponent,
        "num_players": args.num_players,
        "num_episodes_requested": args.num_episodes,
        "num_workers": len(episode_chunks),
        "num_samples": int(arrays["actions"].shape[0]),
        "observation_shape": list(arrays["observations"].shape[1:]),
        "action_shape": list(arrays["actions"].shape[1:]),
        "action_counts": action_counts,
        "button_counts": action_counts if args.action_type == "FILTERED" else None,
    }
    if args.action_type == "HOCKEY_INTENT_DPAD":
        metadata["intent_counts"] = np.bincount(arrays["actions"][:, 0], minlength=len(HOCKEY_INTENT_DPAD_ACTIONS)).astype(int).tolist()
    return arrays, metadata


def main(argv):
    args = parse_cmdline(argv[1:])
    hyperparams = load_hyperparams(
        args.hyperparams,
        required=True,
        base_dir=os.path.dirname(__file__),
    )
    arrays, metadata = collect_demos(args, hyperparams)
    output_path = _make_output_path(args)
    save_demo_shard(output_path, arrays, metadata)
    print(json.dumps({"saved": output_path, **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv)