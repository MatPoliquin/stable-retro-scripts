"""MKII macro action support.

This wrapper exposes a compact Liu Kang-specific action set so PPO can choose
specials directly instead of having to rediscover them through generic button
probes. The default mapping below assumes the standard Genesis 6-button MKII
layout, but the exact in-ROM control assignment still needs to be verified
against your ROM's options screen.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List

import numpy as np

HIGH_PUNCH_BUTTON = "X"
LOW_PUNCH_BUTTON = "Y"
BLOCK_BUTTON = "START"
HIGH_KICK_BUTTON = "A"
LOW_KICK_BUTTON = "B"
RUN_BUTTON = "C"

FIRST_FORWARD_HOLD_FRAMES = 2
DOUBLE_TAP_GAP_FRAMES = 2
SPECIAL_HOLD_FRAMES = 3
POST_SPECIAL_NEUTRAL_FRAMES = 2
BUTTON_HOLD_FRAMES = 5
RUN_HOLD_FRAMES = 3
LONG_CHARGE_FRAMES = 180
BLOCK_HOLD_FRAMES = 4


MACRO_ACTION_NAMES = [
    "NOOP",
    "TOWARD",
    "AWAY",
    "UP",
    "DOWN",
    "BLOCK",
    "RUN",
    "HIGH_PUNCH",
    "LOW_PUNCH",
    "HIGH_KICK",
    "LOW_KICK",
    "HIGH_FIREBALL",
    "LOW_FIREBALL",
    "FLYING_KICK",
    "BICYCLE_KICK",
]


class MK2MacroActionTranslator:
    def __init__(self, button_names: Iterable[str]) -> None:
        self.button_names = [str(name) for name in button_names]
        self.button_index: Dict[str, int] = {
            name.upper(): index for index, name in enumerate(self.button_names)
        }
        self.pending_actions: Deque[np.ndarray] = deque()

    def reset(self) -> None:
        self.pending_actions.clear()

    def get_action_names(self) -> List[str]:
        return list(MACRO_ACTION_NAMES)

    def action_count(self) -> int:
        return len(MACRO_ACTION_NAMES)

    def translate(self, action, info: Dict[str, int] | None) -> np.ndarray:
        if self.pending_actions:
            return self.pending_actions.popleft().copy()

        if self._is_raw_button_action(action):
            return np.asarray(action, dtype=np.int8).copy()

        action_idx = self._coerce_action_index(action)
        if action_idx is None:
            return self._neutral_action()

        facing = self._resolve_relative_directions(info)
        macro = self._build_macro_sequence(action_idx, facing)
        if not macro:
            return self._neutral_action()

        if len(macro) > 1:
            self.pending_actions.extend(frame.copy() for frame in macro[1:])
        return macro[0].copy()

    def _is_raw_button_action(self, action) -> bool:
        action_arr = np.asarray(action)
        return action_arr.ndim > 0 and action_arr.size in {
            len(self.button_names),
            len(self.button_names) * 2,
        }

    def _coerce_action_index(self, action):
        action_arr = np.asarray(action)
        if action_arr.ndim == 0:
            return int(action_arr)
        flat = action_arr.reshape(-1)
        if flat.size == 1:
            return int(flat[0])
        return None

    def _neutral_action(self) -> np.ndarray:
        return np.zeros(len(self.button_names), dtype=np.int8)

    def _repeat_action(self, action: np.ndarray, frames: int) -> List[np.ndarray]:
        return [action.copy() for _ in range(max(0, frames))]

    def _press(self, *button_names: str) -> np.ndarray:
        action = self._neutral_action()
        for button_name in button_names:
            idx = self.button_index.get(str(button_name).upper())
            if idx is not None:
                action[idx] = 1
        return action

    def _direction_action(self, horizontal: str | None = None, vertical: str | None = None, *buttons: str) -> np.ndarray:
        names: List[str] = list(buttons)
        if horizontal:
            names.append(horizontal)
        if vertical:
            names.append(vertical)
        return self._press(*names)

    def _forward_forward_button_sequence(self, direction: str, button_name: str) -> List[np.ndarray]:
        sequence = self._repeat_action(
            self._direction_action(horizontal=direction),
            FIRST_FORWARD_HOLD_FRAMES,
        )
        sequence.extend(self._repeat_action(self._neutral_action(), DOUBLE_TAP_GAP_FRAMES))
        sequence.extend(
            self._repeat_action(
                self._direction_action(direction, None, button_name),
                SPECIAL_HOLD_FRAMES,
            )
        )
        sequence.extend(self._repeat_action(self._neutral_action(), POST_SPECIAL_NEUTRAL_FRAMES))
        return sequence

    def _charge_release_sequence(self, button_name: str, charge_frames: int) -> List[np.ndarray]:
        sequence = self._repeat_action(self._press(button_name), charge_frames)
        sequence.append(self._neutral_action())
        sequence.append(self._neutral_action())
        return sequence

    def _resolve_relative_directions(self, info: Dict[str, int] | None) -> Dict[str, str]:
        if not info:
            return {"toward": "RIGHT", "away": "LEFT"}

        p1_x = int(info.get("x_position", 0))
        p2_x = int(info.get("enemy_x_position", 0))
        if p2_x >= p1_x:
            return {"toward": "RIGHT", "away": "LEFT"}
        return {"toward": "LEFT", "away": "RIGHT"}

    def _build_macro_sequence(self, action_idx: int, facing: Dict[str, str]) -> List[np.ndarray]:
        toward = facing["toward"]
        away = facing["away"]
        action_name = MACRO_ACTION_NAMES[action_idx]

        if action_idx <= 0:
            return [self._neutral_action()]
        if action_name == "TOWARD":
            return [self._direction_action(horizontal=toward)]
        if action_name == "AWAY":
            return [self._direction_action(horizontal=away)]
        if action_name == "UP":
            return [self._direction_action(vertical="UP")]
        if action_name == "DOWN":
            return [self._direction_action(vertical="DOWN")]
        if action_name == "BLOCK":
            sequence = self._repeat_action(self._press(BLOCK_BUTTON), BLOCK_HOLD_FRAMES)
            sequence.append(self._neutral_action())
            return sequence
        if action_name == "RUN":
            sequence = self._repeat_action(self._press(RUN_BUTTON), RUN_HOLD_FRAMES)
            sequence.append(self._neutral_action())
            return sequence
        if action_name == "HIGH_PUNCH":
            sequence = self._repeat_action(self._press(HIGH_PUNCH_BUTTON), BUTTON_HOLD_FRAMES)
            sequence.append(self._neutral_action())
            return sequence
        if action_name == "LOW_PUNCH":
            sequence = self._repeat_action(self._press(LOW_PUNCH_BUTTON), BUTTON_HOLD_FRAMES)
            sequence.append(self._neutral_action())
            return sequence
        if action_name == "HIGH_KICK":
            sequence = self._repeat_action(self._press(HIGH_KICK_BUTTON), BUTTON_HOLD_FRAMES)
            sequence.append(self._neutral_action())
            return sequence
        if action_name == "LOW_KICK":
            sequence = self._repeat_action(self._press(LOW_KICK_BUTTON), BUTTON_HOLD_FRAMES)
            sequence.append(self._neutral_action())
            return sequence
        if action_name == "HIGH_FIREBALL":
            return self._forward_forward_button_sequence(toward, HIGH_PUNCH_BUTTON)
        if action_name == "LOW_FIREBALL":
            return self._forward_forward_button_sequence(toward, LOW_PUNCH_BUTTON)
        if action_name == "FLYING_KICK":
            return self._forward_forward_button_sequence(toward, HIGH_KICK_BUTTON)
        if action_name == "BICYCLE_KICK":
            return self._charge_release_sequence(LOW_KICK_BUTTON, LONG_CHARGE_FRAMES)

        return [self._neutral_action()]