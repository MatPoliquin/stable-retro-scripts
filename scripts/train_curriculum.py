#!/usr/bin/env python3
"""Run multi-phase curriculum training sessions from a JSON definition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import train_live
from utils import load_curriculum


RUNNER_PHASE_METADATA = {
    "name",
    "description",
    "notes",
    "enabled",
    "phase_type",
    "eval_episodes",
}
PATH_KEYS = ("hyperparams", "load_p1_model", "load_opponent_model", "output_basedir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a curriculum of sequential train_live sessions.")
    parser.add_argument("--curriculum", required=True, help="Path to a curriculum JSON file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the curriculum and print resolved phase arguments without training.",
    )
    return parser


def get_train_arg_map() -> Dict[str, argparse.Action]:
    parser = train_live.build_parser()
    return {
        action.dest: action
        for action in parser._actions
        if action.dest and action.dest != "help"
    }


def action_to_option(action: argparse.Action) -> Optional[str]:
    for option in action.option_strings:
        if option.startswith("--"):
            return option
    return None


def resolve_phase_paths(config: Dict[str, Any], curriculum_dir: str) -> Dict[str, Any]:
    resolved = dict(config)
    for key in PATH_KEYS:
        value = resolved.get(key)
        if not value:
            continue
        expanded = os.path.expanduser(str(value))
        if os.path.isabs(expanded):
            resolved[key] = expanded
        else:
            resolved[key] = os.path.abspath(os.path.join(curriculum_dir, expanded))
    return resolved


def merge_phase_config(
    common: Dict[str, Any],
    phase: Dict[str, Any],
    previous_model_path: str,
    curriculum_dir: str,
) -> Dict[str, Any]:
    merged = dict(common)
    for key, value in phase.items():
        if key in RUNNER_PHASE_METADATA:
            continue
        merged[key] = value

    merged = resolve_phase_paths(merged, curriculum_dir)

    if previous_model_path:
        merged["load_p1_model"] = previous_model_path

    return merged


def validate_phase_keys(config: Dict[str, Any], allowed_keys: Iterable[str], phase_label: str) -> None:
    allowed = set(allowed_keys)
    unknown = sorted(key for key in config if key not in allowed)
    if unknown:
        raise ValueError(f"{phase_label} contains unsupported train_live args: {', '.join(unknown)}")


def config_to_cli_args(config: Dict[str, Any], action_map: Dict[str, argparse.Action]) -> List[str]:
    cli_args: List[str] = []

    for key, value in config.items():
        action = action_map[key]
        option = action_to_option(action)
        if option is None or value is None:
            continue

        if isinstance(value, bool):
            if value:
                cli_args.append(option)
            continue

        cli_args.append(f"{option}={value}")

    return cli_args


def build_phase_args(config: Dict[str, Any], curriculum_dir: str, action_map: Dict[str, argparse.Action]) -> argparse.Namespace:
    cli_args = config_to_cli_args(config, action_map)
    args = train_live.parse_cmdline(cli_args)

    for key, value in config.items():
        setattr(args, key, value)

    train_live.prepare_args(args, hyperparams_base_dir=curriculum_dir)
    return args


def format_phase_summary(config: Dict[str, Any]) -> str:
    ordered_keys = (
        "env",
        "state",
        "rf",
        "nn",
        "alg",
        "num_timesteps",
        "num_env",
        "hyperparams",
        "output_basedir",
        "load_p1_model",
    )
    summary = {key: config.get(key) for key in ordered_keys if key in config}
    return json.dumps(summary, indent=2, sort_keys=False)


def run_curriculum(curriculum_path: str, *, dry_run: bool = False) -> Tuple[str, List[Dict[str, str]]]:
    curriculum = load_curriculum(curriculum_path)
    curriculum_dir = curriculum["_base_dir"]
    action_map = get_train_arg_map()
    allowed_keys = action_map.keys()

    common = resolve_phase_paths(curriculum.get("common", {}), curriculum_dir)
    validate_phase_keys(common, allowed_keys, "Curriculum common section")

    curriculum_name = curriculum.get("name") or os.path.basename(curriculum["_resolved_path"])
    print(f"=== Curriculum: {curriculum_name} ===")
    if curriculum.get("description"):
        print(curriculum["description"])

    previous_model_path = ""
    phase_summaries: List[Dict[str, str]] = []
    enabled_phase_count = 0

    for index, phase in enumerate(curriculum["phases"], start=1):
        if not phase.get("enabled", True):
            phase_name = phase.get("name") or f"phase_{index}"
            print(f"Skipping disabled phase {index}: {phase_name}")
            continue

        enabled_phase_count += 1
        phase_name = phase.get("name") or f"phase_{index}"
        phase_type = str(phase.get("phase_type", "train")).lower()
        if phase_type not in {"train", "eval"}:
            raise ValueError(f"Phase '{phase_name}' has unsupported phase_type '{phase_type}'")
        phase_config = merge_phase_config(common, phase, previous_model_path, curriculum_dir)
        validate_phase_keys(phase_config, allowed_keys, f"Phase '{phase_name}'")

        print(f"\n=== Phase {index}: {phase_name} ===")
        if phase.get("description"):
            print(phase["description"])
        print(format_phase_summary(phase_config))

        build_phase_args(phase_config, curriculum_dir, action_map)

        if dry_run:
            continue

        args = build_phase_args(phase_config, curriculum_dir, action_map)
        if phase_type == "eval":
            if not previous_model_path:
                raise ValueError(f"Phase '{phase_name}' is eval-only but no previous model is available.")

            eval_episodes = int(phase.get("eval_episodes", 5))
            result = train_live.run_evaluation_session(
                args,
                previous_model_path,
                eval_episodes=eval_episodes,
            )
            phase_summaries.append(
                {
                    "phase": phase_name,
                    "phase_type": phase_type,
                    "state": str(getattr(args, "state", "")),
                    "model_path": result.model_path,
                    "mean_reward": f"{result.mean_reward:.4f}",
                    "std_reward": f"{result.std_reward:.4f}",
                    "episodes": str(result.episodes),
                }
            )
            print(
                "Evaluation complete:\n"
                f"  state: {args.state}\n"
                f"  model_path: {result.model_path}\n"
                f"  episodes: {result.episodes}\n"
                f"  mean_reward: {result.mean_reward:.4f}\n"
                f"  std_reward: {result.std_reward:.4f}"
            )
        else:
            result = train_live.run_training_session(args)
            previous_model_path = result.final_model_path

            phase_summaries.append(
                {
                    "phase": phase_name,
                    "phase_type": phase_type,
                    "output_dir": result.output_dir,
                    "final_model_path": result.final_model_path,
                    "best_model_path": result.best_model_path,
                }
            )

            print(
                "Phase complete:\n"
                f"  output_dir: {result.output_dir}\n"
                f"  final_model_path: {result.final_model_path}\n"
                f"  best_model_path: {result.best_model_path}"
            )

    if enabled_phase_count == 0:
        raise ValueError("Curriculum has no enabled phases to run.")

    return previous_model_path, phase_summaries


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    final_model_path, phase_summaries = run_curriculum(args.curriculum, dry_run=args.dry_run)

    print("\n=== Curriculum Summary ===")
    if args.dry_run:
        print("Dry run completed; no training sessions were started.")
        return 0

    for summary in phase_summaries:
        if summary.get("phase_type") == "eval":
            print(
                f"- {summary['phase']} (eval):\n"
                f"  state: {summary['state']}\n"
                f"  model_path: {summary['model_path']}\n"
                f"  episodes: {summary['episodes']}\n"
                f"  mean_reward: {summary['mean_reward']}\n"
                f"  std_reward: {summary['std_reward']}"
            )
        else:
            print(
                f"- {summary['phase']}:\n"
                f"  output_dir: {summary['output_dir']}\n"
                f"  final_model_path: {summary['final_model_path']}\n"
                f"  best_model_path: {summary['best_model_path']}"
            )

    print(f"Final chained model: {final_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))