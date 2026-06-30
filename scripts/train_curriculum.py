#!/usr/bin/env python3
"""Run multi-phase curriculum training sessions from a JSON definition."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import play as play_script
import bc_pretrain as bc_pretrain_script
import collect_classic_ai_demos as collect_demos_script
import dagger_loop as dagger_script
import train_live
from utils import load_curriculum, load_hyperparams, resolve_hyperparams_for_model


RUNNER_PHASE_METADATA = {
    "name",
    "description",
    "notes",
    "enabled",
    "phase_type",
    "eval_episodes",
}
PATH_KEYS = ("hyperparams", "load_p1_model", "load_opponent_model", "output_basedir")
COLLECT_DEMOS_PATH_KEYS = ("hyperparams", "output")
BC_PRETRAIN_PATH_KEYS = ("hyperparams", "output_dir", "output_model", "load_model")
DAGGER_PATH_KEYS = ("hyperparams", "output_dir", "model")
POST_PLAY_METADATA = {"enabled"}
POST_PLAY_PATH_KEYS = ("hyperparams", "output_basedir", "model_1", "model_2", "load_p1_model", "load_p2_model")
POST_PLAY_ALLOWED_KEYS = {
    "mode",
    "env",
    "state",
    "num_players",
    "num_env",
    "output_basedir",
    "display_width",
    "display_height",
    "fullscreen",
    "deterministic",
    "single_session",
    "rf",
    "hyperparams",
    "video",
    "video_path",
    "seq_len",
    "max_playback_speed",
    "alg",
    "p1_alg",
    "p2_alg",
    "nn",
    "model1_desc",
    "model2_desc",
    "nnsize",
    "model_1",
    "model_2",
    "load_p1_model",
    "load_p2_model",
    "action_type",
}
SPINNER_FRAMES = ("|", "/", "-", "\\")
ACTIVE_PHASE_COLOR = "\033[1;36m"
RESET_COLOR = "\033[0m"


class CurriculumTerminalDisplay:
    def __init__(self, phase_names: List[str]) -> None:
        self._tty = sys.stdout.isatty()
        self._phase_rows = [
            {
                "name": name,
                "steps": None,
                "reward": None,
                "best_reward": None,
            }
            for name in phase_names
        ]
        self._active_phase_index: Optional[int] = None
        self._spinner_index = 0
        self._games: Optional[int] = None
        self._team1 = {"goals": None, "shots": None, "one_timers": None, "cross_checks": None}
        self._team2 = {"goals": None, "shots": None, "one_timers": None, "cross_checks": None}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._tty or self._thread is not None:
            return

        self.render()
        self._thread = threading.Thread(target=self._spin, name="curriculum-terminal-display", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.render()

    def activate_phase(self, phase_index: int) -> None:
        with self._lock:
            self._active_phase_index = phase_index
            self._spinner_index = 0
            row = self._phase_rows[phase_index]
            if row["steps"] is None:
                row["steps"] = 0
        self.render()

    def complete_phase(self, phase_index: int) -> None:
        with self._lock:
            if self._active_phase_index == phase_index:
                self._active_phase_index = None
        self.render()

    def update_phase(
        self,
        phase_index: int,
        *,
        steps: Optional[int] = None,
        reward: Optional[float] = None,
        best_reward: Optional[float] = None,
    ) -> None:
        with self._lock:
            row = self._phase_rows[phase_index]
            if steps is not None:
                row["steps"] = int(steps)
            if reward is not None:
                row["reward"] = float(reward)
            if best_reward is not None:
                row["best_reward"] = float(best_reward)
        self.render()

    def update_test_totals(self, games: int, team1_stats, team2_stats) -> None:
        with self._lock:
            self._games = int(games)
            self._team1 = {
                "goals": int(getattr(team1_stats, "goals", 0) or 0),
                "shots": int(getattr(team1_stats, "shots", 0) or 0),
                "one_timers": int(getattr(team1_stats, "one_timers", 0) or 0),
                "cross_checks": int(getattr(team1_stats, "cross_checks", 0) or 0),
            }
            self._team2 = {
                "goals": int(getattr(team2_stats, "goals", 0) or 0),
                "shots": int(getattr(team2_stats, "shots", 0) or 0),
                "one_timers": int(getattr(team2_stats, "one_timers", 0) or 0),
                "cross_checks": int(getattr(team2_stats, "cross_checks", 0) or 0),
            }
        self.render()

    def clear_test_totals(self) -> None:
        with self._lock:
            self._games = None
            self._team1 = {"goals": None, "shots": None, "one_timers": None, "cross_checks": None}
            self._team2 = {"goals": None, "shots": None, "one_timers": None, "cross_checks": None}
        self.render()

    def render(self) -> None:
        with self._lock:
            rows = [dict(row) for row in self._phase_rows]
            active_phase_index = self._active_phase_index
            spinner_index = self._spinner_index
            games = self._games
            team1 = dict(self._team1)
            team2 = dict(self._team2)

        lines: List[str] = []
        for index, row in enumerate(rows, start=1):
            prefix = f"{index}."
            phase_name = row["name"]
            if active_phase_index == index - 1:
                spinner = SPINNER_FRAMES[spinner_index]
                prefix = f"{ACTIVE_PHASE_COLOR}{spinner}{RESET_COLOR}" if self._tty else spinner
                phase_name = f"{ACTIVE_PHASE_COLOR}{phase_name}{RESET_COLOR}" if self._tty else phase_name

            lines.append(
                f"{prefix} {phase_name}: steps={self._format_steps(row['steps'])} "
                f"reward={self._format_reward(row['reward'])} best_reward={self._format_reward(row['best_reward'])}"
            )

        lines.extend(
            [
                "",
                "===============",
                "TEST",
                "===============",
                f"number of games: {games if games is not None else 'N/A'}",
                self._format_team_line("team1", team1),
                self._format_team_line("team2", team2),
            ]
        )

        output = "\n".join(lines)
        if self._tty:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(output + "\n")
            sys.stdout.flush()
            return

        print(output, flush=True)

    def _spin(self) -> None:
        while not self._stop_event.wait(0.2):
            with self._lock:
                if self._active_phase_index is None:
                    continue
                self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
            self.render()

    @staticmethod
    def _format_steps(value: Optional[int]) -> str:
        return "--" if value is None else str(int(value))

    @staticmethod
    def _format_reward(value: Optional[float]) -> str:
        return "--" if value is None else f"{value:.2f}"

    @staticmethod
    def _format_team_line(label: str, stats: Dict[str, Optional[int]]) -> str:
        def value_or_dash(value: Optional[int]) -> str:
            return "--" if value is None else str(int(value))

        return (
            f"{label}: goals={value_or_dash(stats['goals'])} "
            f"shots={value_or_dash(stats['shots'])} "
            f"one-timers={value_or_dash(stats['one_timers'])} "
            f"cross checks={value_or_dash(stats['cross_checks'])}"
        )


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
    return get_arg_map_from_parser(parser)


def get_collect_arg_map() -> Dict[str, argparse.Action]:
    parser = collect_demos_script.build_parser()
    return get_arg_map_from_parser(parser)


def get_bc_arg_map() -> Dict[str, argparse.Action]:
    parser = bc_pretrain_script.build_parser()
    return get_arg_map_from_parser(parser)


def get_dagger_arg_map() -> Dict[str, argparse.Action]:
    parser = dagger_script.build_parser()
    return get_arg_map_from_parser(parser)


def get_arg_map_from_parser(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
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
    return resolve_paths_for_keys(config, curriculum_dir, PATH_KEYS)


def resolve_paths_for_keys(config: Dict[str, Any], curriculum_dir: str, path_keys: Iterable[str]) -> Dict[str, Any]:
    resolved = dict(config)
    for key in path_keys:
        value = resolved.get(key)
        if not value:
            continue
        expanded = os.path.expanduser(str(value))
        if os.path.isabs(expanded):
            resolved[key] = expanded
        else:
            resolved[key] = os.path.abspath(os.path.join(curriculum_dir, expanded))
    return resolved


def resolve_dataset_paths(paths: Iterable[str], curriculum_dir: str) -> List[str]:
    resolved_paths: List[str] = []
    for path in paths:
        expanded = os.path.expanduser(str(path))
        if os.path.isabs(expanded):
            resolved_paths.append(expanded)
        else:
            resolved_paths.append(os.path.abspath(os.path.join(curriculum_dir, expanded)))
    return resolved_paths


def resolve_post_play_paths(config: Dict[str, Any], curriculum_dir: str) -> Dict[str, Any]:
    resolved = dict(config)
    for key in POST_PLAY_PATH_KEYS:
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


def merge_utility_phase_config(
    common: Dict[str, Any],
    phase: Dict[str, Any],
    curriculum_dir: str,
    path_keys: Iterable[str],
) -> Dict[str, Any]:
    merged = dict(common)
    for key, value in phase.items():
        if key in RUNNER_PHASE_METADATA:
            continue
        merged[key] = value

    merged = resolve_paths_for_keys(merged, curriculum_dir, path_keys)
    if "datasets" in merged and merged["datasets"]:
        datasets = merged["datasets"]
        if isinstance(datasets, str):
            datasets = [datasets]
        merged["datasets"] = resolve_dataset_paths(datasets, curriculum_dir)
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

        if isinstance(value, list):
            cli_args.append(option)
            cli_args.extend(str(item) for item in value)
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


def build_collect_args(config: Dict[str, Any], action_map: Dict[str, argparse.Action]) -> argparse.Namespace:
    cli_args = config_to_cli_args(config, action_map)
    args = collect_demos_script.parse_cmdline(cli_args)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def build_bc_args(config: Dict[str, Any], action_map: Dict[str, argparse.Action]) -> argparse.Namespace:
    cli_args = config_to_cli_args(config, action_map)
    args = bc_pretrain_script.parse_cmdline(cli_args)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def build_dagger_args(config: Dict[str, Any], action_map: Dict[str, argparse.Action]) -> argparse.Namespace:
    cli_args = config_to_cli_args(config, action_map)
    args = dagger_script.parse_cmdline(cli_args)
    for key, value in config.items():
        setattr(args, key, value)
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
        "load_opponent_model",
    )
    summary = {key: config.get(key) for key in ordered_keys if key in config}
    return json.dumps(summary, indent=2, sort_keys=False)


def format_utility_summary(config: Dict[str, Any]) -> str:
    ordered_keys = (
        "env",
        "state",
        "rf",
        "nn",
        "num_episodes",
        "num_workers",
        "max_steps",
        "shard_name",
        "datasets",
        "epochs",
        "batch_size",
        "rare_weight",
        "positive_weight_max",
        "release_weight",
        "output",
        "output_dir",
        "output_model",
        "model",
        "base_datasets",
        "rounds",
        "episodes_per_round",
        "bc_epochs",
        "bc_batch_size",
        "bc_learning_rate",
        "hyperparams",
    )
    summary = {key: config.get(key) for key in ordered_keys if key in config}
    return json.dumps(summary, indent=2, sort_keys=False)


def validate_post_play_keys(config: Dict[str, Any]) -> None:
    allowed = POST_PLAY_ALLOWED_KEYS | POST_PLAY_METADATA
    unknown = sorted(key for key in config if key not in allowed)
    if unknown:
        raise ValueError(f"Curriculum post_play contains unsupported play args: {', '.join(unknown)}")


def build_post_play_config(
    curriculum: Dict[str, Any],
    last_model_config: Optional[Dict[str, Any]],
    final_model_path: str,
    curriculum_dir: str,
) -> Optional[Dict[str, Any]]:
    post_play = curriculum.get("post_play")
    if post_play is None:
        return None
    if not isinstance(post_play, dict):
        raise TypeError("Curriculum 'post_play' must be a JSON object.")
    if not post_play.get("enabled", True):
        return None
    if not last_model_config:
        raise ValueError("Curriculum post_play requires at least one model-producing phase.")

    validate_post_play_keys(post_play)

    config = {key: value for key, value in last_model_config.items() if key in POST_PLAY_ALLOWED_KEYS}
    config.update({key: value for key, value in post_play.items() if key not in POST_PLAY_METADATA})
    config.setdefault("mode", "model_vs_game")
    config.setdefault("rf", "PostPlay")
    config.setdefault("single_session", False)
    config["num_env"] = int(post_play.get("num_env", 1))
    config["model_1"] = final_model_path

    return resolve_post_play_paths(config, curriculum_dir)


def post_play_to_cli_args(config: Dict[str, Any]) -> List[str]:
    cli_args: List[str] = []

    for key, value in config.items():
        if key not in POST_PLAY_ALLOWED_KEYS or value is None:
            continue

        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(option)
            continue

        cli_args.append(f"{option}={value}")

    return cli_args


def build_post_play_args(config: Dict[str, Any]) -> argparse.Namespace:
    cli_args = post_play_to_cli_args(config)
    args = play_script.parse_cmdline(cli_args)

    for key, value in config.items():
        setattr(args, key, value)

    args.hyperparams_dict = resolve_hyperparams_for_model(
        load_hyperparams(
            args.hyperparams,
            required=True,
            base_dir=os.path.dirname(play_script.__file__),
        ),
        args.nn,
    )
    return args


def format_post_play_summary(config: Dict[str, Any], final_model_path: str) -> str:
    ordered_keys = (
        "mode",
        "env",
        "state",
        "rf",
        "single_session",
        "nn",
        "alg",
        "num_players",
        "num_env",
        "hyperparams",
    )
    summary = {key: config.get(key) for key in ordered_keys if key in config}
    summary["model_1"] = final_model_path
    return json.dumps(summary, indent=2, sort_keys=False)


def get_enabled_phase_names(phases: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for index, phase in enumerate(phases, start=1):
        if not phase.get("enabled", True):
            continue
        names.append(phase.get("name") or f"phase_{index}")
    return names


def run_curriculum(curriculum_path: str, *, dry_run: bool = False) -> Tuple[str, List[Dict[str, str]]]:
    curriculum = load_curriculum(curriculum_path)
    curriculum_dir = curriculum["_base_dir"]
    action_map = get_train_arg_map()
    collect_action_map = get_collect_arg_map()
    bc_action_map = get_bc_arg_map()
    dagger_action_map = get_dagger_arg_map()
    allowed_keys = action_map.keys()
    collect_allowed_keys = collect_action_map.keys()
    bc_allowed_keys = bc_action_map.keys()
    dagger_allowed_keys = dagger_action_map.keys()

    common = dict(curriculum.get("common", {}))

    curriculum_name = curriculum.get("name") or os.path.basename(curriculum["_resolved_path"])
    print(f"=== Curriculum: {curriculum_name} ===")
    if curriculum.get("description"):
        print(curriculum["description"])

    previous_model_path = ""
    last_model_config: Optional[Dict[str, Any]] = None
    collected_dataset_paths: List[str] = []
    phase_summaries: List[Dict[str, str]] = []
    enabled_phase_count = 0
    terminal_display = CurriculumTerminalDisplay(get_enabled_phase_names(curriculum["phases"])) if not dry_run else None

    if terminal_display is not None:
        terminal_display.start()

    try:
        for index, phase in enumerate(curriculum["phases"], start=1):
            if not phase.get("enabled", True):
                phase_name = phase.get("name") or f"phase_{index}"
                print(f"Skipping disabled phase {index}: {phase_name}")
                continue

            enabled_phase_count += 1
            display_index = enabled_phase_count - 1
            phase_name = phase.get("name") or f"phase_{index}"
            phase_type = str(phase.get("phase_type", "train")).lower()
            if phase_type == "test":
                phase_type = "eval"
            if phase_type in {"collect", "demos", "collect_demo", "collect_demos"}:
                phase_type = "collect_demos"
            if phase_type in {"bc", "behavior_cloning", "bc_pretrain"}:
                phase_type = "bc_pretrain"
            if phase_type in {"dagger", "classic_ai_dagger"}:
                phase_type = "dagger"
            if phase_type not in {"train", "eval", "collect_demos", "bc_pretrain", "dagger"}:
                raise ValueError(f"Phase '{phase_name}' has unsupported phase_type '{phase_type}'")

            if phase_type in {"train", "eval"}:
                phase_config = merge_phase_config(common, phase, previous_model_path, curriculum_dir)
                validate_phase_keys(phase_config, allowed_keys, f"Phase '{phase_name}'")
            elif phase_type == "collect_demos":
                phase_config = merge_utility_phase_config(common, phase, curriculum_dir, COLLECT_DEMOS_PATH_KEYS)
                validate_phase_keys(phase_config, collect_allowed_keys, f"Phase '{phase_name}'")
            elif phase_type == "bc_pretrain":
                phase_config = merge_utility_phase_config(common, phase, curriculum_dir, BC_PRETRAIN_PATH_KEYS)
                if not phase_config.get("datasets"):
                    if not collected_dataset_paths:
                        raise ValueError(f"Phase '{phase_name}' is bc_pretrain but no datasets are available.")
                    phase_config["datasets"] = list(collected_dataset_paths)
                validate_phase_keys(phase_config, bc_allowed_keys, f"Phase '{phase_name}'")
            else:
                phase_config = merge_utility_phase_config(common, phase, curriculum_dir, DAGGER_PATH_KEYS)
                if "base_datasets" in phase_config and phase_config["base_datasets"]:
                    base_datasets = phase_config["base_datasets"]
                    if isinstance(base_datasets, str):
                        base_datasets = [base_datasets]
                    phase_config["base_datasets"] = resolve_dataset_paths(base_datasets, curriculum_dir)
                elif collected_dataset_paths:
                    phase_config["base_datasets"] = list(collected_dataset_paths)
                else:
                    raise ValueError(f"Phase '{phase_name}' is dagger but no base_datasets are available.")
                if not phase_config.get("model"):
                    if not previous_model_path:
                        raise ValueError(f"Phase '{phase_name}' is dagger but no previous model is available.")
                    phase_config["model"] = previous_model_path
                validate_phase_keys(phase_config, dagger_allowed_keys, f"Phase '{phase_name}'")

            print(f"\n=== Phase {index}: {phase_name} ===")
            if phase.get("description"):
                print(phase["description"])
            if phase_type in {"train", "eval"}:
                print(format_phase_summary(phase_config))
                build_phase_args(phase_config, curriculum_dir, action_map)
            else:
                print(format_utility_summary(phase_config))
                if phase_type == "collect_demos":
                    build_collect_args(phase_config, collect_action_map)
                elif phase_type == "bc_pretrain":
                    build_bc_args(phase_config, bc_action_map)
                else:
                    build_dagger_args(phase_config, dagger_action_map)

            if phase_type == "train":
                last_model_config = phase_config
            elif phase_type == "bc_pretrain":
                last_model_config = phase_config
            elif phase_type == "dagger":
                last_model_config = phase_config

            if dry_run:
                if phase_type == "collect_demos":
                    args = build_collect_args(phase_config, collect_action_map)
                    collected_dataset_paths.append(collect_demos_script._make_output_path(args))
                elif phase_type == "bc_pretrain":
                    args = build_bc_args(phase_config, bc_action_map)
                    previous_model_path = bc_pretrain_script._make_output_model_path(args)
                    last_model_config = phase_config
                elif phase_type == "dagger":
                    args = build_dagger_args(phase_config, dagger_action_map)
                    if not args.skip_retrain:
                        previous_model_path = os.path.join(
                            os.path.expanduser(args.output_dir),
                            f"classic_ai_dagger_round_{int(args.rounds):02d}.zip",
                        )
                    last_model_config = phase_config
                continue

            if phase_type == "collect_demos":
                args = build_collect_args(phase_config, collect_action_map)
                if terminal_display is not None:
                    terminal_display.activate_phase(display_index)
                    terminal_display.clear_test_totals()
                hyperparams = resolve_hyperparams_for_model(
                    load_hyperparams(
                        args.hyperparams,
                        required=True,
                        base_dir=os.path.dirname(collect_demos_script.__file__),
                    ),
                    args.nn,
                )
                arrays, metadata = collect_demos_script.collect_demos(args, hyperparams)
                dataset_path = collect_demos_script._make_output_path(args)
                collect_demos_script.save_demo_shard(dataset_path, arrays, metadata)
                collected_dataset_paths.append(dataset_path)
                if terminal_display is not None:
                    terminal_display.update_phase(display_index, steps=int(metadata.get("num_samples", 0)))
                    terminal_display.complete_phase(display_index)
                phase_summaries.append(
                    {
                        "phase": phase_name,
                        "phase_type": phase_type,
                        "dataset_path": dataset_path,
                        "num_samples": str(metadata.get("num_samples", "")),
                    }
                )
                print(
                    "Demo collection complete:\n"
                    f"  dataset_path: {dataset_path}\n"
                    f"  num_samples: {metadata.get('num_samples')}"
                )
                continue

            if phase_type == "bc_pretrain":
                args = build_bc_args(phase_config, bc_action_map)
                if terminal_display is not None:
                    terminal_display.activate_phase(display_index)
                    terminal_display.clear_test_totals()
                hyperparams = resolve_hyperparams_for_model(
                    load_hyperparams(
                        args.hyperparams,
                        required=True,
                        base_dir=os.path.dirname(bc_pretrain_script.__file__),
                    ),
                    args.nn,
                )
                model_path = bc_pretrain_script.train_bc(args, hyperparams)
                previous_model_path = model_path
                last_model_config = phase_config
                if terminal_display is not None:
                    terminal_display.complete_phase(display_index)
                phase_summaries.append(
                    {
                        "phase": phase_name,
                        "phase_type": phase_type,
                        "final_model_path": model_path,
                    }
                )
                print("BC pretraining complete:\n" f"  final_model_path: {model_path}")
                continue

            if phase_type == "dagger":
                args = build_dagger_args(phase_config, dagger_action_map)
                if terminal_display is not None:
                    terminal_display.activate_phase(display_index)
                    terminal_display.clear_test_totals()
                hyperparams = resolve_hyperparams_for_model(
                    load_hyperparams(
                        args.hyperparams,
                        required=True,
                        base_dir=os.path.dirname(dagger_script.__file__),
                    ),
                    args.nn,
                )
                model_path = dagger_script.run_dagger(args, hyperparams)
                previous_model_path = model_path
                last_model_config = phase_config
                if terminal_display is not None:
                    terminal_display.complete_phase(display_index)
                phase_summaries.append(
                    {
                        "phase": phase_name,
                        "phase_type": phase_type,
                        "final_model_path": model_path,
                    }
                )
                print("DAgger complete:\n" f"  final_model_path: {model_path}")
                continue

            args = build_phase_args(phase_config, curriculum_dir, action_map)
            if phase_type == "eval":
                if not previous_model_path:
                    raise ValueError(f"Phase '{phase_name}' is eval-only but no previous model is available.")

                if terminal_display is not None:
                    terminal_display.activate_phase(display_index)
                    terminal_display.update_test_totals(0, train_live.LiveTeamTotals(), train_live.LiveTeamTotals())

                def report_test_progress(
                    games_completed: int,
                    mean_reward: float,
                    team1_stats,
                    team2_stats,
                    *,
                    phase_slot: int = display_index,
                ) -> None:
                    if terminal_display is None:
                        return
                    terminal_display.update_phase(phase_slot, reward=mean_reward)
                    terminal_display.update_test_totals(games_completed, team1_stats, team2_stats)

                eval_episodes = int(phase.get("eval_episodes", 5))
                result = train_live.run_evaluation_session(
                    args,
                    previous_model_path,
                    eval_episodes=eval_episodes,
                    status_reporter=report_test_progress,
                )
                if terminal_display is not None:
                    terminal_display.update_test_totals(result.episodes, result.team1_totals, result.team2_totals)
                    terminal_display.complete_phase(display_index)
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
                args.alg_verbose = False
                if terminal_display is not None:
                    terminal_display.activate_phase(display_index)
                    terminal_display.clear_test_totals()

                def report_progress(
                    steps: int,
                    reward: float,
                    best_reward: float,
                    games: int,
                    team1_stats,
                    team2_stats,
                    *,
                    phase_slot: int = display_index,
                ) -> None:
                    if terminal_display is None:
                        return
                    terminal_display.update_phase(
                        phase_slot,
                        steps=steps,
                        reward=reward,
                        best_reward=best_reward,
                    )

                result = train_live.run_training_session(args, status_reporter=report_progress)
                if terminal_display is not None:
                    terminal_display.complete_phase(display_index)
                previous_model_path = result.final_model_path
                last_model_config = phase_config

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
    finally:
        if terminal_display is not None:
            terminal_display.stop()

    if enabled_phase_count == 0:
        raise ValueError("Curriculum has no enabled phases to run.")

    final_model_for_play = previous_model_path if previous_model_path else "__curriculum_final_model__.zip"
    post_play_config = build_post_play_config(
        curriculum,
        last_model_config,
        final_model_for_play,
        curriculum_dir,
    )
    if post_play_config is not None:
        print("\n=== Post-Training Play ===")
        print(format_post_play_summary(post_play_config, final_model_for_play))
        build_post_play_args(post_play_config)

        if not dry_run:
            play_script.main(["play.py", *post_play_to_cli_args(post_play_config)])

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
        elif summary.get("phase_type") == "collect_demos":
            print(
                f"- {summary['phase']} (collect_demos):\n"
                f"  dataset_path: {summary['dataset_path']}\n"
                f"  num_samples: {summary['num_samples']}"
            )
        elif summary.get("phase_type") == "bc_pretrain":
            print(
                f"- {summary['phase']} (bc_pretrain):\n"
                f"  final_model_path: {summary['final_model_path']}"
            )
        elif summary.get("phase_type") == "dagger":
            print(
                f"- {summary['phase']} (dagger):\n"
                f"  final_model_path: {summary['final_model_path']}"
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