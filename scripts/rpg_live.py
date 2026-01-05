#!/usr/bin/env python3

"""LLM-driven live RPG player for stable-retro.

Displays the game frame in a pygame window (left) and the LLM's latest
"thinking" + chosen buttons (right). The LLM is fed:
  - the game image
  - a small curated subset of named info variables (from the game's data.json)

Backends:
  - Ollama (default): POST /api/chat
  - OpenAI-compatible (LM Studio): POST /v1/chat/completions

Example:
  python3 scripts/rpg_live.py --game PokemonRed-GameBoy --model qwen2.5vl:7b

Notes:
  - Ensure the ROM is imported into stable-retro before running.
  - This script does not send raw RAM bytes.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import pygame.freetype
import cv2

import stable_retro as retro


DEFAULT_VAR_CANDIDATES: List[str] = [
    # Common / generic
    "current_map",
    "map_location",
    "current_level",
    "player_x",
    "player_y",
    "link_x",
    "link_y",
    "link_direction",
    # Progress / inventory-ish
    "badges",
    "rupees",
    "bombs",
    "keys",
    "triforce_pieces",
    # Party snapshot
    "party_count",
    "party1_species",
    "party1_level",
    "party1_current_hp",
    "party1_max_hp",
]


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}

    # Last resort: repr for unknown types
    return repr(value)


def select_info_vars(info: Dict[str, Any], requested: Optional[Sequence[str]], max_vars: int) -> Dict[str, Any]:
    if not info:
        return {}

    if requested:
        out: Dict[str, Any] = {}
        for key in requested:
            if key in info:
                out[key] = _coerce_jsonable(info[key])
        return out

    out = {}
    for key in DEFAULT_VAR_CANDIDATES:
        if key in info:
            out[key] = _coerce_jsonable(info[key])
            if len(out) >= max_vars:
                break
    return out


def pygame_surface_from_rgb(frame_rgb: np.ndarray) -> pygame.Surface:
    # frame_rgb is HxWx3
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got shape={frame_rgb.shape}")

    # pygame.surfarray.make_surface expects WxHx3
    return pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))


def encode_surface_png_base64(surface: pygame.Surface) -> str:
    # Ensure we send a real PNG (pygame saves to BMP when writing to file-like objects).
    # Convert surface -> numpy RGB -> PNG in-memory using OpenCV.
    arr = pygame.surfarray.array3d(surface)  # (w, h, 3)
    arr = np.transpose(arr, (1, 0, 2))       # -> (h, w, 3)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode frame to PNG")
    return base64.b64encode(buf).decode("ascii")


def wrap_text(font: pygame.freetype.Font, text: str, max_width: int) -> List[str]:
    words = (text or "").replace("\r", "").split()
    if not words:
        return [""]

    lines: List[str] = []
    current: List[str] = []

    for word in words:
        trial = (" ".join(current + [word])).strip()
        rect = font.get_rect(trial)
        if rect.width <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


@dataclass
class LlmResult:
    thinking: str
    buttons: List[str]
    raw_text: str


class LlmClient:
    def __init__(
        self,
        backend: str,
        base_url: str,
        model: str,
        timeout_s: float,
    ) -> None:
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat_vision(self, prompt: str, image_b64_png: str) -> str:
        if self.backend == "ollama":
            return self._chat_ollama(prompt, image_b64_png)
        if self.backend == "openai":
            return self._chat_openai(prompt, image_b64_png)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    def _chat_ollama(self, prompt: str, image_b64_png: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64_png],
                }
            ],
        }
        resp = self._post_json(url, payload)
        message = resp.get("message") or {}
        return str(message.get("content") or "")

    def _chat_openai(self, prompt: str, image_b64_png: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64_png}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        resp = self._post_json(url, payload)
        choices = resp.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        return str(message.get("content") or "")


def parse_llm_response(text: str) -> LlmResult:
    raw = text or ""

    # Preferred: strict JSON object.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            thinking = str(obj.get("thinking") or obj.get("thoughts") or "")
            buttons = obj.get("buttons") or obj.get("press") or []
            if isinstance(buttons, str):
                buttons = [b.strip() for b in buttons.split(",") if b.strip()]
            if isinstance(buttons, list):
                buttons = [str(b).strip() for b in buttons if str(b).strip()]
            else:
                buttons = []
            return LlmResult(thinking=thinking, buttons=buttons, raw_text=raw)
    except Exception:
        pass

    # Fallback: try to find a "Buttons:" line.
    thinking = raw.strip()
    buttons: List[str] = []
    for line in raw.splitlines():
        if line.lower().startswith("buttons"):
            _, _, rest = line.partition(":")
            buttons = [b.strip() for b in rest.split(",") if b.strip()]
            break

    return LlmResult(thinking=thinking, buttons=buttons, raw_text=raw)


def build_prompt(
    game: str,
    buttons: Sequence[str],
    selected_vars: Dict[str, Any],
) -> str:
    allowed = ", ".join(buttons)
    vars_json = json.dumps(selected_vars, ensure_ascii=False)

    return (
        "You are playing a retro RPG.\n"
        "Goal: make sensible progress (explore, navigate menus, talk, fight when needed).\n"
        "Return STRICT JSON only, with keys:\n"
        "  thinking: short text\n"
        "  buttons: array of button names to press this frame\n"
        "Allowed button names: "
        + allowed
        + "\n"
        + "Game: "
        + game
        + "\n"
        + "Known variables (from data.json): "
        + vars_json
        + "\n"
        + "Now choose buttons."
    )


def buttons_to_action(buttons: Sequence[str], pressed: Sequence[str]) -> np.ndarray:
    name_to_idx = {b: i for i, b in enumerate(buttons)}
    action = np.zeros((len(buttons),), dtype=np.uint8)

    for b in pressed:
        key = b.strip().upper()
        if key in name_to_idx:
            action[name_to_idx[key]] = 1

    return action


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LLM plays stable-retro RPGs with a live UI")

    parser.add_argument("--game", required=True, help="Game name, e.g. PokemonRed-GameBoy")
    parser.add_argument("--state", default=retro.State.DEFAULT, help="State name (default from metadata)")
    parser.add_argument("--scenario", default=None, help="Scenario (default: scenario.json)")
    parser.add_argument(
        "--integration",
        default="stable",
        choices=["stable", "experimental", "contrib", "all"],
        help="stable-retro integration set",
    )

    parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "openai"],
        help="LLM backend: ollama (/api/chat) or OpenAI-compatible (/v1/chat/completions)",
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="LLM server base URL")
    parser.add_argument("--model", required=True, help="Model name as known by the backend")
    parser.add_argument("--timeout", type=float, default=30.0, help="LLM request timeout (seconds)")

    parser.add_argument(
        "--vars",
        default="",
        help="Comma-separated info vars to send (data.json-derived). If empty, uses a small heuristic set.",
    )
    parser.add_argument("--max-vars", type=int, default=12, help="Max auto-selected vars to send")

    parser.add_argument("--fps", type=int, default=30, help="UI FPS cap")
    parser.add_argument("--decision-interval", type=int, default=6, help="LLM call every N env steps")

    args = parser.parse_args(argv)

    integration_map = {
        "stable": retro.data.Integrations.STABLE,
        "experimental": retro.data.Integrations.EXPERIMENTAL,
        "contrib": retro.data.Integrations.CONTRIB,
        "all": retro.data.Integrations.ALL,
    }

    requested_vars = [v.strip() for v in args.vars.split(",") if v.strip()]

    try:
        env = retro.make(
            args.game,
            state=args.state,
            scenario=args.scenario,
            inttype=integration_map[args.integration],
            obs_type=retro.Observations.IMAGE,
            render_mode="rgb_array",
        )
    except FileNotFoundError:
        # Convenience: many RPG datasets live under "experimental".
        if args.integration == "stable":
            env = retro.make(
                args.game,
                state=args.state,
                scenario=args.scenario,
                inttype=retro.data.Integrations.EXPERIMENTAL,
                obs_type=retro.Observations.IMAGE,
                render_mode="rgb_array",
            )
        else:
            raise

    buttons = [str(b).upper() for b in getattr(env, "buttons", [])]
    if not buttons:
        raise RuntimeError("Env has no button list; cannot map LLM output")

    llm = LlmClient(
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
    )

    pygame.init()
    pygame.display.set_caption(f"rpg_live - {args.game}")

    font = pygame.freetype.SysFont("symbol", 18)
    font.antialiased = True
    header_font = pygame.freetype.SysFont("symbol", 22)
    header_font.antialiased = True

    obs, info0 = env.reset()
    info: Dict[str, Any] = dict(info0 or {})

    frame = env.render()
    if frame is None:
        # fall back to obs if env returns obs as image
        frame = obs

    game_surface = pygame_surface_from_rgb(frame)

    left_w, left_h = game_surface.get_width(), game_surface.get_height()
    scale = 3
    left_panel_w = left_w * scale
    left_panel_h = left_h * scale
    right_panel_w = 560

    screen = pygame.display.set_mode((left_panel_w + right_panel_w, left_panel_h))
    clock = pygame.time.Clock()

    last_llm = LlmResult(thinking="", buttons=[], raw_text="")
    last_action = np.zeros((len(buttons),), dtype=np.uint8)
    last_prompt_vars: Dict[str, Any] = {}

    step_idx = 0
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if paused:
            # still render UI
            clock.tick(args.fps)
        else:
            if step_idx % max(1, args.decision_interval) == 0:
                # Use latest rendered frame for LLM.
                llm_frame = env.render()
                if llm_frame is None:
                    llm_frame = frame

                llm_surface = pygame_surface_from_rgb(llm_frame)
                image_b64 = encode_surface_png_base64(llm_surface)

                last_prompt_vars = select_info_vars(
                    info,
                    requested=requested_vars if requested_vars else None,
                    max_vars=max(0, args.max_vars),
                )

                prompt = build_prompt(args.game, buttons, last_prompt_vars)

                try:
                    text = llm.chat_vision(prompt, image_b64)
                    last_llm = parse_llm_response(text)
                except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
                    last_llm = LlmResult(
                        thinking=f"LLM error: {exc}",
                        buttons=[],
                        raw_text="",
                    )

                last_action = buttons_to_action(buttons, last_llm.buttons)

            step_result = env.step(last_action)
            if len(step_result) == 4:
                obs, _, done, info = step_result
                terminated = bool(done)
                truncated = False
            else:
                obs, _, terminated, truncated, info = step_result

            info = dict(info or {})

            frame = env.render()
            if frame is None:
                frame = obs

            if terminated or truncated:
                obs, info0 = env.reset()
                info = dict(info0 or {})
                frame = env.render() or obs

            step_idx += 1
            clock.tick(args.fps)

        # --- draw ---
        screen.fill((20, 20, 24))

        # Left: game frame
        try:
            game_surface = pygame_surface_from_rgb(frame)
            scaled = pygame.transform.scale(game_surface, (left_panel_w, left_panel_h))
            screen.blit(scaled, (0, 0))
        except Exception:
            # If frame is invalid, keep window alive
            pass

        # Right: LLM panel
        x0 = left_panel_w
        pygame.draw.rect(screen, (28, 28, 34), pygame.Rect(x0, 0, right_panel_w, left_panel_h))
        pygame.draw.line(screen, (60, 60, 70), (x0, 0), (x0, left_panel_h), 2)

        y = 10
        header_font.render_to(screen, (x0 + 12, y), "LLM", (230, 230, 240))
        y += 30

        status = "PAUSED" if paused else f"step={step_idx} interval={args.decision_interval}"
        font.render_to(screen, (x0 + 12, y), status, (180, 180, 200))
        y += 26

        # Buttons pressed
        pressed_names = [buttons[i] for i, v in enumerate(last_action) if int(v) == 1]
        font.render_to(screen, (x0 + 12, y), "Buttons:", (200, 200, 220))
        y += 20
        for line in wrap_text(font, ", ".join(pressed_names) if pressed_names else "(none)", right_panel_w - 24):
            font.render_to(screen, (x0 + 12, y), line, (235, 235, 255))
            y += 18
        y += 10

        # Vars
        font.render_to(screen, (x0 + 12, y), "Vars:", (200, 200, 220))
        y += 20
        vars_text = json.dumps(last_prompt_vars, ensure_ascii=False)
        for line in wrap_text(font, vars_text, right_panel_w - 24):
            font.render_to(screen, (x0 + 12, y), line, (210, 210, 230))
            y += 18
        y += 10

        # Thinking
        font.render_to(screen, (x0 + 12, y), "Thinking:", (200, 200, 220))
        y += 20
        thinking_text = last_llm.thinking or ""
        for line in wrap_text(font, thinking_text, right_panel_w - 24):
            if y > left_panel_h - 24:
                break
            font.render_to(screen, (x0 + 12, y), line, (235, 235, 255))
            y += 18

        pygame.display.flip()

    env.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
