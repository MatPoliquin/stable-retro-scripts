import json
import os
from typing import Any, Dict, Optional


def resolve_config_path(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    if not path:
        return None

    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded

    if base_dir:
        candidate = os.path.abspath(os.path.join(base_dir, expanded))
        if os.path.isfile(candidate):
            return candidate

    return os.path.abspath(expanded)


def load_hyperparams(path: Optional[str], *, required: bool = True, base_dir: Optional[str] = None) -> Dict[str, Any]:
    if not path:
        if required:
            raise ValueError("Hyperparameters path is required but was not provided.")
        return {}

    resolved = resolve_config_path(path, base_dir=base_dir)
    if resolved is None:
        if required:
            raise ValueError("Hyperparameters path is required but was not provided.")
        return {}

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Hyperparameters file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as handle:
        hyperparams = json.load(handle)

    if not isinstance(hyperparams, dict):
        raise TypeError(f"Hyperparameters file must contain a JSON object: {resolved}")

    return hyperparams


def resolve_clip_reward(args, hyperparams: Optional[Dict[str, Any]]) -> bool:
    if hasattr(args, "clip_reward") and getattr(args, "clip_reward") is not None:
        return bool(getattr(args, "clip_reward"))

    if hyperparams and "clip_reward" in hyperparams:
        return bool(hyperparams.get("clip_reward"))

    return True
