#!/usr/bin/env python3
"""Compare two models by training them and recording a side-by-side video."""

import argparse
import copy
import io
import os
from contextlib import redirect_stdout
from typing import List, Tuple

# Hide the pygame prompt before importing pygame
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2  # type: ignore
import numpy as np  # type: ignore
import pygame  # type: ignore
import pygame.freetype  # type: ignore
from stable_baselines3 import PPO  # type: ignore

from common import init_logger
from env_utils import init_env, get_button_names
from models_utils import get_num_parameters, get_model_probabilities
from torchsummary import summary
import game_wrappers_mgr as games
import train


DEFAULT_OUTPUT_DIR = os.path.expanduser("~/OUTPUT/compare_models")
DEFAULT_FPS = 60
DEFAULT_DURATION_SECONDS = 120
DEFAULT_HYPERPARAMS = os.path.join(os.path.dirname(__file__), "..", "hyperparams", "default.json")
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
DEFAULT_MARGIN = 20
DEFAULT_FOOTER_HEIGHT = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two models and record a side-by-side comparison video.")

    parser.add_argument("--env", type=str, default="NHL941on1-Genesis",
                        help="Retro environment identifier (default: NHL941on1-Genesis)")
    parser.add_argument("--state", type=str, default=None,
                        help="Specific environment state to load (default: None)")
    parser.add_argument("--action-type", type=str, default="FILTERED",
                        choices=["FILTERED", "DISCRETE", "MULTI_DISCRETE"],
                        help="Action space type to use (default: FILTERED)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Number of environment timesteps to train each model (default: 1_000_000)")
    parser.add_argument("--train-num-env", type=int, default=8,
                        help="Number of parallel environments to use during training (default: 8)")
    parser.add_argument("--model1-policy", type=str, default="CnnPolicy",
                        help="Policy class name for model 1 (default: CnnPolicy)")
    parser.add_argument("--model2-policy", type=str, default="MlpPolicy",
                        help="Policy class name for model 2 (default: MlpPolicy)")
    parser.add_argument("--model1-hyperparams", type=str, default=DEFAULT_HYPERPARAMS,
                        help="Path to hyperparameter JSON for model 1")
    parser.add_argument("--model2-hyperparams", type=str, default=DEFAULT_HYPERPARAMS,
                        help="Path to hyperparameter JSON for model 2")
    parser.add_argument("--video-output", type=str, default="model_comparison.mp4",
                        help="Output path for the comparison video (default: model_comparison.mp4)")
    parser.add_argument("--video-duration", type=float, default=DEFAULT_DURATION_SECONDS,
                        help="Length of the recorded video in seconds (default: 120)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="Frames per second for the recorded video (default: 60)")
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_WIDTH,
                        help="Width of the comparison window (default: 1920)")
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_HEIGHT,
                        help="Height of the comparison window (default: 1080)")
    parser.add_argument("--display-height", type=int, dest="window_height",
                        help="Deprecated alias for --window-height")
    parser.add_argument("--output-basedir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Base directory for training artifacts (default: ~/OUTPUT/compare_models)")
    parser.add_argument("--algorithm", type=str, default="ppo2",
                        help="Training algorithm identifier (PPO, PPO2, ES - case-insensitive; default: ppo2)")

    parser.add_argument("--model1-label", type=str, default=None,
                        help="Custom label to display under model 1 (default: policy name)")
    parser.add_argument("--model2-label", type=str, default=None,
                        help="Custom label to display under model 2 (default: policy name)")
    parser.add_argument("--summary-duration", type=float, default=20.0,
                        help="Duration in seconds of the pre-roll model summary segment (default: 20)")

    return parser.parse_args()


def ensure_hyperparams(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {abs_path}")
    return abs_path


def normalize_algorithm(name: str) -> str:
    if not name:
        return "ppo2"

    lowered = name.strip().lower()

    if lowered in {"ppo", "ppo2"}:
        return "ppo2"
    if lowered in {"es"}:
        return "es"

    raise ValueError(
        f"Unsupported algorithm '{name}'. Supported values are: PPO, PPO2, ES."
    )


def build_train_args(base: argparse.Namespace, policy: str, hyperparams_path: str,
                      run_id: str) -> argparse.Namespace:
    cli_args = [
        f"--alg={base.algorithm}",
        f"--nn={policy}",
        f"--env={base.env}",
        f"--num_timesteps={base.timesteps}",
        f"--num_env={base.train_num_env}",
        f"--output_basedir={os.path.join(base.output_basedir, run_id)}",
        f"--hyperparams={hyperparams_path}",
        f"--action_type={base.action_type}",
        f"--display_width={base.window_width}",
        f"--display_height={base.window_height}",
    ]

    if base.state:
        cli_args.append(f"--state={base.state}")

    parsed = train.parse_cmdline(cli_args)
    parsed.model1_desc = base.model1_label or policy
    parsed.model2_desc = base.model2_label or policy
    parsed.num_players = 1
    parsed.play = False
    parsed.selfplay = False

    return parsed


def train_model(train_args: argparse.Namespace) -> Tuple[str, argparse.Namespace]:
    os.makedirs(os.path.expanduser(train_args.output_basedir), exist_ok=True)
    logger = init_logger(train_args)
    trainer = train.ModelTrainer(train_args, logger)
    model_save_root = trainer.train()
    trainer.env.close()
    if hasattr(trainer.p1_model, "env") and trainer.p1_model.env is not None:
        trainer.p1_model.env.close()
    model_path = f"{model_save_root}.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected trained model at {model_path}, but it was not created.")
    return model_path, train_args


def get_env_frame(env) -> np.ndarray:
    frame = None

    # First try the vectorized render method (works for most VecEnvs)
    try:
        frame = env.render(mode="rgb_array")
        if isinstance(frame, list):
            frame = next((img for img in frame if img is not None), None)
    except TypeError:
        frame = None

    # Some VecEnvs expose get_images instead of render returns
    if frame is None and hasattr(env, "get_images"):
        try:
            images = env.get_images()
            if images:
                frame = next((img for img in images if img is not None), None)
        except (NotImplementedError, TypeError):
            frame = None

    # Fallback to env_method without additional kwargs
    if frame is None:
        try:
            frames = env.env_method("render")
            if isinstance(frames, list):
                frame = next((img for img in frames if img is not None), None)
            else:
                frame = frames
        except (TypeError, AttributeError):
            frame = None

    if frame is None:
        raise RuntimeError("Environment did not return an rgb_array frame. Make sure the env supports rendering.")

    return frame


def prepare_action_info(probabilities, actions, button_count: int) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float).flatten()
    acts = np.asarray(actions, dtype=float).flatten()

    if probs.size < button_count:
        if probs.size == 0:
            probs = np.zeros(button_count, dtype=float)
        else:
            probs = np.pad(probs, (0, button_count - probs.size), constant_values=0.0)
    elif probs.size > button_count:
        probs = probs[:button_count]

    if acts.size < button_count:
        if acts.size == 0:
            acts = np.zeros(button_count, dtype=float)
        else:
            acts = np.pad(acts, (0, button_count - acts.size), constant_values=0.0)
    elif acts.size > button_count:
        acts = acts[:button_count]

    return probs, acts


def generate_model_summary(model: PPO, env, policy_name: str) -> str:
    obs_space = getattr(model, "observation_space", None) or env.observation_space

    try:
        if policy_name in ("MlpPolicy", "EntityAttentionPolicy", "CustomMlpPolicy", "MlpDropoutPolicy", "CombinedPolicy", "AttentionMLPPolicy", "HockeyMultiHeadPolicy"):
            if hasattr(obs_space, "shape") and obs_space.shape:
                input_shape = (1, obs_space.shape[0])
            else:
                return f"Model summary unavailable for policy {policy_name}: unsupported observation space"
        elif policy_name in ("CnnPolicy", "ImpalaCnnPolicy", "CustomCnnPolicy", "CnnTransformerPolicy", "ViTPolicy", "DartPolicy"):
            if hasattr(obs_space, "shape") and len(obs_space.shape) >= 3:
                input_shape = (obs_space.shape[-3], obs_space.shape[-2], obs_space.shape[-1])
            else:
                return f"Model summary unavailable for policy {policy_name}: unsupported observation space"
        else:
            return f"Model summary unavailable for policy {policy_name}"

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary(model.policy, input_shape)
        return buffer.getvalue()
    except Exception as exc:  # pylint: disable=broad-except
        return f"Model summary unavailable: {exc}"


def init_play_env(train_args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, any]:
    play_args = copy.deepcopy(train_args)
    play_args.num_env = 1
    play_args.num_players = 1
    play_args.output_basedir = None
    play_args.alg_verbose = False
    play_args.info_verbose = False

    games.wrappers.init(play_args)
    env = init_env(None, 1, play_args.state, play_args.num_players, play_args,
                   use_sticky_action=False, use_display=False, use_frame_skip=True)
    obs = env.reset()
    frame = get_env_frame(env)
    return obs, frame, env


def load_trained_model(model_path: str, env) -> PPO:
    model = PPO.load(model_path, env=env)
    return model


def play_and_record(model1: PPO, env1, obs1: np.ndarray, label1: str, params1: int,
                    model2: PPO, env2, obs2: np.ndarray, label2: str, params2: int,
                    button_names: List[str],
                    summary_text1: str, summary_text2: str, summary_duration: float,
                    initial_frame1: np.ndarray, initial_frame2: np.ndarray,
                    video_path: str, duration_seconds: float, fps: int,
                    window_width: int, window_height: int) -> None:
    pygame.init()
    pygame.freetype.init()

    frame_height, frame_width = initial_frame1.shape[:2]
    aspect_ratio = frame_width / frame_height if frame_height else 4 / 3

    margin = DEFAULT_MARGIN
    footer_height = DEFAULT_FOOTER_HEIGHT

    available_width = window_width - (margin * 3)
    available_height = window_height - (margin * 2) - footer_height

    target_width = available_width // 2
    target_height = int(target_width / aspect_ratio)

    if target_height > available_height:
        target_height = available_height
        target_width = int(target_height * aspect_ratio)

    left_x = margin
    right_x = window_width - margin - target_width

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Model Comparison")

    font_vs = pygame.freetype.SysFont("Arial", 64)
    font_label = pygame.freetype.SysFont("Arial", 36)
    font_detail = pygame.freetype.SysFont("Arial", 28)
    font_timer = pygame.freetype.SysFont("Arial", 24)
    font_summary = pygame.freetype.SysFont("Courier New", 14)

    params_text1 = f"Parameters: {params1:,}"
    params_text2 = f"Parameters: {params2:,}"

    def draw_button_info(x_pos: int, start_y: int, names, probs, actions) -> int:
        y_pos = start_y
        for name, prob, action in zip(names, probs, actions):
            is_pressed = action >= 0.5
            status = "ON " if is_pressed else "off"
            color = (255, 200, 120) if is_pressed else (190, 190, 190)
            text = f"{name:<8} {prob:>5.2f} {status}"
            font_detail.render_to(screen, (x_pos, y_pos), text, color)
            y_pos += 24
        return y_pos

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (window_width, window_height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {video_path}")

    total_frames = int(duration_seconds * fps)
    clock = pygame.time.Clock()

    summary_frames = max(0, int(summary_duration * fps))
    summary_line_height = font_summary.get_sized_height() + 2
    max_summary_lines = max(1, (window_height - 2 * margin - footer_height - 80) // summary_line_height)

    def prepare_summary_lines(text: str) -> List[str]:
        raw_lines = [line.rstrip() for line in text.strip().splitlines()]
        if not raw_lines:
            raw_lines = ["Summary unavailable"]
        if len(raw_lines) > max_summary_lines:
            truncated = raw_lines[:max_summary_lines - 1]
            truncated.append("... (truncated) ...")
            return truncated
        return raw_lines

    summary_lines1 = prepare_summary_lines(summary_text1)
    summary_lines2 = prepare_summary_lines(summary_text2)

    if summary_frames > 0:
        for frame_idx in range(summary_frames):
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    break

            if quit_requested:
                writer.release()
                pygame.quit()
                return

            screen.fill((15, 15, 15))

            heading_offset = margin
            font_label.render_to(screen, (left_x, heading_offset), f"{label1} Summary", (200, 200, 200))
            font_label.render_to(screen, (right_x, heading_offset), f"{label2} Summary", (200, 200, 200))

            text_y_left = heading_offset + 40
            for line in summary_lines1:
                font_summary.render_to(screen, (left_x, text_y_left), line, (210, 210, 210))
                text_y_left += summary_line_height

            text_y_right = heading_offset + 40
            for line in summary_lines2:
                font_summary.render_to(screen, (right_x, text_y_right), line, (210, 210, 210))
                text_y_right += summary_line_height

            remaining_time = max(summary_duration - frame_idx / fps, 0.0)
            timer_text = f"Gameplay starts in {remaining_time:0.1f}s"
            font_timer.render_to(screen, (window_width - margin - 280, window_height - footer_height // 2),
                                 timer_text, (180, 180, 180))

            pygame.display.flip()

            frame_pixels = pygame.surfarray.array3d(screen)
            frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
            writer.write(cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR))
            clock.tick(fps)

    frame1 = initial_frame1
    frame2 = initial_frame2

    total_reward1 = 0.0
    total_reward2 = 0.0

    for frame_idx in range(total_frames):
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
                break

        if quit_requested:
            break

        try:
            prob1_vals = np.asarray(get_model_probabilities(model1, obs1), dtype=float).flatten()
        except (RuntimeError, AttributeError, ValueError, TypeError):
            prob1_vals = np.array([])
        actions1, _ = model1.predict(obs1, deterministic=True)
        actions1_flat = np.asarray(actions1, dtype=float).flatten()
        obs1, reward1_arr, done1, _ = env1.step(actions1)
        reward1_value = float(np.array(reward1_arr).reshape(-1)[0])
        total_reward1 += reward1_value
        done1_flag = bool(np.any(done1))
        if done1_flag:
            obs1 = env1.reset()
        frame1 = get_env_frame(env1)

        try:
            prob2_vals = np.asarray(get_model_probabilities(model2, obs2), dtype=float).flatten()
        except (RuntimeError, AttributeError, ValueError, TypeError):
            prob2_vals = np.array([])
        actions2, _ = model2.predict(obs2, deterministic=True)
        actions2_flat = np.asarray(actions2, dtype=float).flatten()
        obs2, reward2_arr, done2, _ = env2.step(actions2)
        reward2_value = float(np.array(reward2_arr).reshape(-1)[0])
        total_reward2 += reward2_value
        done2_flag = bool(np.any(done2))
        if done2_flag:
            obs2 = env2.reset()
        frame2 = get_env_frame(env2)

        display_probs1, display_actions1 = prepare_action_info(prob1_vals, actions1_flat, len(button_names))
        display_probs2, display_actions2 = prepare_action_info(prob2_vals, actions2_flat, len(button_names))

        screen.fill((15, 15, 15))

        resized1 = cv2.resize(frame1, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized2 = cv2.resize(frame2, (target_width, target_height), interpolation=cv2.INTER_AREA)

        surf1 = pygame.surfarray.make_surface(np.transpose(resized1, (1, 0, 2)))
        surf2 = pygame.surfarray.make_surface(np.transpose(resized2, (1, 0, 2)))

        screen.blit(surf1, (left_x, margin))
        screen.blit(surf2, (right_x, margin))

        vs_pos = (window_width // 2 - 40, margin + target_height // 2 - 40)
        font_vs.render_to(screen, vs_pos, "VS", (255, 255, 255))

        label_y = margin + target_height + 20
        font_label.render_to(screen, (left_x, label_y), label1, (200, 200, 200))
        font_label.render_to(screen, (right_x, label_y), label2, (200, 200, 200))

        params_y = label_y + 35
        font_detail.render_to(screen, (left_x, params_y), params_text1, (180, 220, 255))
        font_detail.render_to(screen, (right_x, params_y), params_text2, (180, 220, 255))

        reward_y = params_y + 30
        font_detail.render_to(screen, (left_x, reward_y), f"Total Reward: {total_reward1:.2f}", (220, 220, 120))
        font_detail.render_to(screen, (right_x, reward_y), f"Total Reward: {total_reward2:.2f}", (220, 220, 120))

        buttons_start_y = reward_y + 30
        draw_button_info(left_x, buttons_start_y, button_names, display_probs1, display_actions1)
        draw_button_info(right_x, buttons_start_y, button_names, display_probs2, display_actions2)

        timer_text = f"{frame_idx / fps:.1f}s / {duration_seconds:.1f}s"
        font_timer.render_to(screen, (window_width - margin - 200, window_height - footer_height // 2),
                             timer_text, (180, 180, 180))

        divider_x = window_width // 2
        pygame.draw.line(screen, (80, 80, 80),
                         (divider_x, margin),
                         (divider_x, margin + target_height), width=4)

        pygame.display.flip()

        frame_pixels = pygame.surfarray.array3d(screen)
        frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
        writer.write(cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR))
        clock.tick(fps)

        if done1_flag:
            total_reward1 = 0.0
        if done2_flag:
            total_reward2 = 0.0

    writer.release()
    pygame.quit()


def main() -> None:
    args = parse_args()

    try:
        args.algorithm = normalize_algorithm(args.algorithm)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.model1_hyperparams = ensure_hyperparams(args.model1_hyperparams)
    args.model2_hyperparams = ensure_hyperparams(args.model2_hyperparams)

    os.makedirs(args.output_basedir, exist_ok=True)

    print("--- Training Model 1 ---")
    train_args1 = build_train_args(args, args.model1_policy, args.model1_hyperparams, "model1")
    model_path1, trained_args1 = train_model(train_args1)
    label1 = args.model1_label or args.model1_policy

    print("--- Training Model 2 ---")
    train_args2 = build_train_args(args, args.model2_policy, args.model2_hyperparams, "model2")
    model_path2, trained_args2 = train_model(train_args2)
    label2 = args.model2_label or args.model2_policy

    print("--- Preparing environments ---")
    obs1, initial_frame1, env1 = init_play_env(trained_args1)
    obs2, initial_frame2, env2 = init_play_env(trained_args2)

    button_names = list(get_button_names(trained_args1))

    model1 = load_trained_model(model_path1, env1)
    model2 = load_trained_model(model_path2, env2)

    summary_text1 = generate_model_summary(model1, env1, trained_args1.nn)
    summary_text2 = generate_model_summary(model2, env2, trained_args2.nn)

    params1 = get_num_parameters(model1)
    params2 = get_num_parameters(model2)

    print(f"--- Recording video to {args.video_output} ---")
    try:
        play_and_record(
            model1,
            env1,
            obs1,
            label1,
            params1,
            model2,
            env2,
            obs2,
            label2,
            params2,
            button_names,
            summary_text1,
            summary_text2,
            args.summary_duration,
            initial_frame1,
            initial_frame2,
            args.video_output,
            args.video_duration,
            args.fps,
            args.window_width,
            args.window_height,
        )
    finally:
        env1.close()
        env2.close()

    print("Comparison video created successfully.")


if __name__ == "__main__":
    main()
