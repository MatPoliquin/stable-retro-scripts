#!/usr/bin/env python3
"""Compare two models by training them and recording a side-by-side video."""

import argparse
import copy
import io
import json
import math
import os
import time
from contextlib import redirect_stdout
from typing import List, Optional, Tuple

# Hide the pygame prompt before importing pygame
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2  # type: ignore
import numpy as np  # type: ignore
import pygame  # type: ignore
import pygame.freetype  # type: ignore
from stable_baselines3 import PPO  # type: ignore
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore

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
DEFAULT_FONT = "Arial"


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
    parser.add_argument("--hyperparams-duration", type=float, default=10.0,
                        help="Duration in seconds of the pre-roll hyperparameters segment (default: 10)")
    parser.add_argument("--reward-curve-duration", type=float, default=12.0,
                        help="Duration in seconds of the pre-roll reward curve segment (default: 12)")

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


def train_model(train_args: argparse.Namespace) -> Tuple[str, argparse.Namespace, str]:
    os.makedirs(os.path.expanduser(train_args.output_basedir), exist_ok=True)
    logger = init_logger(train_args)
    log_dir = getattr(logger, "dir", None)
    trainer = train.ModelTrainer(train_args, logger)
    model_save_root = trainer.train()
    trainer.env.close()
    if hasattr(trainer.p1_model, "env") and trainer.p1_model.env is not None:
        trainer.p1_model.env.close()
    model_path = f"{model_save_root}.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected trained model at {model_path}, but it was not created.")
    return model_path, train_args, log_dir or trainer.output_fullpath


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


def load_hyperparams_text(hyperparams_path: str) -> str:
    try:
        with open(hyperparams_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return json.dumps(data, indent=2)
    except Exception as exc:  # pylint: disable=broad-except
        return f"Unable to load hyperparameters: {exc}"


def load_reward_series(log_dir: str, tag: str = "rollout/ep_rew_mean") -> List[Tuple[float, float]]:
    if not log_dir or not os.path.isdir(log_dir):
        return []

    try:
        accumulator = EventAccumulator(log_dir, size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        if tag not in scalar_tags:
            return []
        events = accumulator.Scalars(tag)
        return [(event.step, event.value) for event in events]
    except Exception:  # pylint: disable=broad-except
        return []


def create_reward_curve_surface(
    window_size: Tuple[int, int],
    rewards_a: List[Tuple[float, float]],
    rewards_b: List[Tuple[float, float]],
    label_a: str,
    label_b: str,
) -> Optional["pygame.Surface"]:
    if not rewards_a and not rewards_b:
        return None

    pygame.font.init()
    plot_width, plot_height = window_size
    surface = pygame.Surface(window_size)
    surface.fill((0, 0, 0))

    title_font = pygame.font.SysFont(DEFAULT_FONT, 48)
    subtitle_font = pygame.font.SysFont(DEFAULT_FONT, 32)

    title = title_font.render("Training Reward Curve", True, (255, 255, 255))
    surface.blit(title, title.get_rect(center=(plot_width // 2, 60)))

    subtitle = subtitle_font.render("Mean episode reward per training step", True, (180, 180, 180))
    surface.blit(subtitle, subtitle.get_rect(center=(plot_width // 2, 120)))

    margin = 120
    graph_rect = pygame.Rect(
        margin,
        margin + 80,
        plot_width - 2 * margin,
        plot_height - (margin + 80) - margin,
    )
    pygame.draw.rect(surface, (60, 60, 60), graph_rect, width=2)

    plotted_any = False

    all_rewards = [pt[1] for pt in rewards_a] + [pt[1] for pt in rewards_b]
    all_steps = [pt[0] for pt in rewards_a] + [pt[0] for pt in rewards_b]

    reward_min = min(all_rewards) if all_rewards else 0.0
    reward_max = max(all_rewards) if all_rewards else 1.0
    step_min = min(all_steps) if all_steps else 0.0
    step_max = max(all_steps) if all_steps else 1.0

    def draw_series(points: List[Tuple[float, float]], color: Tuple[int, int, int]) -> None:
        nonlocal plotted_any
        if len(points) < 2:
            return
        steps = [pt[0] for pt in points]
        rewards = [pt[1] for pt in points]
        min_reward = min(rewards)
        max_reward = max(rewards)
        if math.isclose(max_reward, min_reward):
            max_reward += 1.0
            min_reward -= 1.0

        step_min = min(steps)
        step_max = max(steps)
        if math.isclose(step_max, step_min):
            step_max += 1.0

        normalized_points = []
        for step, reward in points:
            x = graph_rect.left + (step - step_min) / (step_max - step_min) * graph_rect.width
            y = graph_rect.bottom - (reward - min_reward) / (max_reward - min_reward) * graph_rect.height
            normalized_points.append((x, y))

        pygame.draw.lines(surface, color, False, normalized_points, width=3)
        plotted_any = True

    draw_series(rewards_a, (255, 99, 71))
    draw_series(rewards_b, (30, 144, 255))

    legend_font = pygame.font.SysFont(DEFAULT_FONT, 28)
    legend_y = graph_rect.top - 40
    legend_padding = 20
    line_width = 40

    def draw_legend(color: Tuple[int, int, int], text: str, x_offset: int) -> int:
        pygame.draw.line(
            surface,
            color,
            (graph_rect.left + x_offset, legend_y),
            (graph_rect.left + x_offset + line_width, legend_y),
            width=6,
        )
        label = legend_font.render(text, True, (220, 220, 220))
        label_rect = label.get_rect()
        label_rect.topleft = (graph_rect.left + x_offset + line_width + 10, legend_y - label_rect.height // 2)
        surface.blit(label, label_rect)
        return x_offset + line_width + label_rect.width + legend_padding + 10

    x_offset = 0
    if rewards_a:
        x_offset = draw_legend((255, 99, 71), label_a, x_offset)
    if rewards_b:
        draw_legend((30, 144, 255), label_b, x_offset)

    axis_font = pygame.font.SysFont(DEFAULT_FONT, 28)
    axis_label_font = pygame.font.SysFont(DEFAULT_FONT, 32)

    has_step_range = bool(all_steps) and step_max > step_min
    has_reward_range = bool(all_rewards) and reward_max > reward_min

    # X-axis ticks
    if has_step_range:
        num_ticks = 5
        step_interval = (step_max - step_min) / (num_ticks - 1)
        for idx in range(num_ticks):
            step_value = step_min + idx * step_interval
            x = graph_rect.left + (step_value - step_min) / (step_max - step_min) * graph_rect.width
            pygame.draw.line(surface, (90, 90, 90), (x, graph_rect.bottom), (x, graph_rect.bottom + 10), 2)
            tick_text = axis_font.render(f"{int(step_value):,}", True, (200, 200, 200))
            text_rect = tick_text.get_rect()
            text_rect.center = (x, graph_rect.bottom + 30)
            surface.blit(tick_text, text_rect)
    elif all_steps:
        x = graph_rect.centerx
        pygame.draw.line(surface, (90, 90, 90), (x, graph_rect.bottom), (x, graph_rect.bottom + 10), 2)
        tick_text = axis_font.render(f"{int(step_min):,}", True, (200, 200, 200))
        text_rect = tick_text.get_rect()
        text_rect.center = (x, graph_rect.bottom + 30)
        surface.blit(tick_text, text_rect)

    # Y-axis ticks
    if has_reward_range:
        num_ticks = 5
        reward_interval = (reward_max - reward_min) / (num_ticks - 1)
        for idx in range(num_ticks):
            reward_value = reward_min + idx * reward_interval
            y = graph_rect.bottom - (reward_value - reward_min) / (reward_max - reward_min) * graph_rect.height
            pygame.draw.line(surface, (90, 90, 90), (graph_rect.left - 10, y), (graph_rect.left, y), 2)
            tick_text = axis_font.render(f"{reward_value:.1f}", True, (200, 200, 200))
            text_rect = tick_text.get_rect()
            text_rect.right = graph_rect.left - 15
            text_rect.centery = y
            surface.blit(tick_text, text_rect)
    elif all_rewards:
        y = graph_rect.centery
        pygame.draw.line(surface, (90, 90, 90), (graph_rect.left - 10, y), (graph_rect.left, y), 2)
        tick_text = axis_font.render(f"{reward_min:.1f}", True, (200, 200, 200))
        text_rect = tick_text.get_rect()
        text_rect.right = graph_rect.left - 15
        text_rect.centery = y
        surface.blit(tick_text, text_rect)

    # Axis labels
    x_axis_label = axis_label_font.render("Timesteps", True, (220, 220, 220))
    x_rect = x_axis_label.get_rect()
    x_rect.center = (graph_rect.centerx, graph_rect.bottom + 60)
    surface.blit(x_axis_label, x_rect)

    y_axis_label = axis_label_font.render("Reward", True, (220, 220, 220))
    y_axis_label = pygame.transform.rotate(y_axis_label, 90)
    y_rect = y_axis_label.get_rect()
    y_rect.center = (graph_rect.left - 70, graph_rect.centery)
    surface.blit(y_axis_label, y_rect)

    if not plotted_any:
        info_font = pygame.font.SysFont(DEFAULT_FONT, 32)
        message = "Not enough reward data to plot a curve"
        info_surface = info_font.render(message, True, (200, 200, 200))
        surface.blit(info_surface, info_surface.get_rect(center=graph_rect.center))

    return surface


def init_play_env(train_args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, any]:
    play_args = copy.deepcopy(train_args)
    play_args.num_env = 1
    play_args.num_players = 1
    play_args.output_basedir = None
    play_args.alg_verbose = False
    play_args.info_verbose = False

    games.wrappers.init(play_args)
    env = init_env(None, 1, play_args.state, play_args.num_players, play_args,
                   use_sticky_action=False, use_display=False, use_frame_skip=False)
    obs = env.reset()
    frame = get_env_frame(env)
    return obs, frame, env


def load_trained_model(model_path: str, env) -> PPO:
    model = PPO.load(model_path, env=env)
    return model


def play_and_record(model1: PPO, env1, obs1: np.ndarray, label1: str, params1: int,
                    model2: PPO, env2, obs2: np.ndarray, label2: str, params2: int,
                    button_names: List[str],
                    reward_series1: List[Tuple[float, float]],
                    reward_series2: List[Tuple[float, float]],
                    reward_curve_duration: float,
                    hyperparams_text1: str, hyperparams_text2: str, hyperparams_duration: float,
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
    font_mono = pygame.freetype.SysFont("Courier New", 14)

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

    if fps <= 0:
        raise ValueError("FPS must be a positive integer")

    clock = pygame.time.Clock()

    requested_summary_frames = max(0, int(summary_duration * fps))
    summary_line_height = font_mono.get_sized_height() + 2
    max_summary_lines = max(1, (window_height - 2 * margin - footer_height - 80) // summary_line_height)
    requested_hyperparams_frames = max(0, int(hyperparams_duration * fps))
    hyperparams_line_height = summary_line_height

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
    hyperparams_lines1 = prepare_summary_lines(hyperparams_text1)
    hyperparams_lines2 = prepare_summary_lines(hyperparams_text2)

    reward_curve_surface = create_reward_curve_surface(
        (window_width, window_height), reward_series1, reward_series2, label1, label2
    )
    reward_curve_frames = max(0, int(reward_curve_duration * fps))

    def render_text_block(lines_left: List[str], lines_right: List[str], heading_left: str, heading_right: str) -> None:
        screen.fill((15, 15, 15))

        heading_offset = margin
        font_label.render_to(screen, (left_x, heading_offset), heading_left, (200, 200, 200))
        font_label.render_to(screen, (right_x, heading_offset), heading_right, (200, 200, 200))

        text_y_left = heading_offset + 40
        for line in lines_left:
            font_mono.render_to(screen, (left_x, text_y_left), line, (210, 210, 210))
            text_y_left += hyperparams_line_height

        text_y_right = heading_offset + 40
        for line in lines_right:
            font_mono.render_to(screen, (right_x, text_y_right), line, (210, 210, 210))
            text_y_right += hyperparams_line_height

    if reward_curve_surface is not None and reward_curve_frames > 0:
        for frame_idx in range(reward_curve_frames):
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    break

            if quit_requested:
                writer.release()
                pygame.quit()
                return

            screen.blit(reward_curve_surface, (0, 0))

            remaining_time = max(reward_curve_duration - frame_idx / fps, 0.0)
            timer_text = f"Hyperparameters in {remaining_time:0.1f}s"
            font_timer.render_to(
                screen,
                (window_width - margin - 320, window_height - footer_height // 2),
                timer_text,
                (180, 180, 180),
            )

            pygame.display.flip()

            frame_pixels = pygame.surfarray.array3d(screen)
            frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
            writer.write(cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR))
            clock.tick(fps)

    hyperparams_frames = requested_hyperparams_frames

    if hyperparams_frames > 0:
        for frame_idx in range(hyperparams_frames):
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    break

            if quit_requested:
                writer.release()
                pygame.quit()
                return

            render_text_block(hyperparams_lines1, hyperparams_lines2,
                              f"{label1} Hyperparameters", f"{label2} Hyperparameters")

            remaining_time = max(hyperparams_duration - frame_idx / fps, 0.0)
            timer_text = f"Model summary in {remaining_time:0.1f}s"
            font_timer.render_to(screen, (window_width - margin - 310, window_height - footer_height // 2),
                                 timer_text, (180, 180, 180))

            pygame.display.flip()

            frame_pixels = pygame.surfarray.array3d(screen)
            frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
            writer.write(cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR))
            clock.tick(fps)

    summary_frames = requested_summary_frames

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

            render_text_block(summary_lines1, summary_lines2,
                              f"{label1} Summary", f"{label2} Summary")

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

    gameplay_frames = max(0, int(duration_seconds * fps))
    gameplay_start_time = time.time()
    action_repeat = 4
    repeat_counter = 0
    current_action1 = None
    current_action2 = None
    prob1_vals = np.array([])
    prob2_vals = np.array([])
    actions1_flat = np.array([])
    actions2_flat = np.array([])
    display_probs1 = np.zeros(len(button_names))
    display_probs2 = np.zeros(len(button_names))
    display_actions1 = np.zeros(len(button_names))
    display_actions2 = np.zeros(len(button_names))

    for frame_idx in range(gameplay_frames):
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
                break

        if quit_requested:
            break

        if repeat_counter <= 0:
            try:
                prob1_vals = np.asarray(get_model_probabilities(model1, obs1), dtype=float).flatten()
            except (RuntimeError, AttributeError, ValueError, TypeError):
                prob1_vals = np.array([])
            current_action1, _ = model1.predict(obs1, deterministic=True)
            actions1_flat = np.asarray(current_action1, dtype=float).flatten()

            try:
                prob2_vals = np.asarray(get_model_probabilities(model2, obs2), dtype=float).flatten()
            except (RuntimeError, AttributeError, ValueError, TypeError):
                prob2_vals = np.array([])
            current_action2, _ = model2.predict(obs2, deterministic=True)
            actions2_flat = np.asarray(current_action2, dtype=float).flatten()

            display_probs1, display_actions1 = prepare_action_info(prob1_vals, actions1_flat, len(button_names))
            display_probs2, display_actions2 = prepare_action_info(prob2_vals, actions2_flat, len(button_names))
            repeat_counter = action_repeat

        obs1, reward1_arr, done1, _ = env1.step(current_action1)
        reward1_value = float(np.array(reward1_arr).reshape(-1)[0])
        total_reward1 += reward1_value
        done1_flag = bool(np.any(done1))
        if done1_flag:
            obs1 = env1.reset()
            repeat_counter = 0
        frame1 = get_env_frame(env1)

        obs2, reward2_arr, done2, _ = env2.step(current_action2)
        reward2_value = float(np.array(reward2_arr).reshape(-1)[0])
        total_reward2 += reward2_value
        done2_flag = bool(np.any(done2))
        if done2_flag:
            obs2 = env2.reset()
            repeat_counter = 0
        frame2 = get_env_frame(env2)

        repeat_counter = max(0, repeat_counter - 1)

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

        wall_elapsed = time.time() - gameplay_start_time
        frame_elapsed = frame_idx / fps
        elapsed_seconds = max(frame_elapsed, wall_elapsed)
        if duration_seconds > 0 and wall_elapsed >= duration_seconds:
            break

        elapsed_seconds = min(elapsed_seconds, duration_seconds)
        timer_text = f"{elapsed_seconds:.1f}s / {duration_seconds:.1f}s"
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
    model_path1, trained_args1, log_dir1 = train_model(train_args1)
    label1 = args.model1_label or args.model1_policy

    print("--- Training Model 2 ---")
    train_args2 = build_train_args(args, args.model2_policy, args.model2_hyperparams, "model2")
    model_path2, trained_args2, log_dir2 = train_model(train_args2)
    label2 = args.model2_label or args.model2_policy

    print("--- Preparing environments ---")
    obs1, initial_frame1, env1 = init_play_env(trained_args1)
    obs2, initial_frame2, env2 = init_play_env(trained_args2)

    button_names = list(get_button_names(trained_args1))

    model1 = load_trained_model(model_path1, env1)
    model2 = load_trained_model(model_path2, env2)

    summary_text1 = generate_model_summary(model1, env1, trained_args1.nn)
    summary_text2 = generate_model_summary(model2, env2, trained_args2.nn)
    hyperparams_text1 = load_hyperparams_text(trained_args1.hyperparams)
    hyperparams_text2 = load_hyperparams_text(trained_args2.hyperparams)
    reward_series1 = load_reward_series(log_dir1)
    reward_series2 = load_reward_series(log_dir2)

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
            reward_series1,
            reward_series2,
            args.reward_curve_duration,
            hyperparams_text1,
            hyperparams_text2,
            args.hyperparams_duration,
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
