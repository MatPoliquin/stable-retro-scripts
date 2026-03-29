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


def load_json_dict(path: Optional[str], *, required: bool = True, base_dir: Optional[str] = None, label: str = "JSON") -> Dict[str, Any]:
    if not path:
        if required:
            raise ValueError(f"{label} path is required but was not provided.")
        return {}

    resolved = resolve_config_path(path, base_dir=base_dir)
    if resolved is None:
        if required:
            raise ValueError(f"{label} path is required but was not provided.")
        return {}

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"{label} file must contain a JSON object: {resolved}")

    return payload


def load_curriculum(path: str, *, base_dir: Optional[str] = None) -> Dict[str, Any]:
    resolved = resolve_config_path(path, base_dir=base_dir)
    if resolved is None:
        raise ValueError("Curriculum path is required but was not provided.")

    curriculum = load_json_dict(resolved, label="Curriculum")

    phases = curriculum.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"Curriculum must define a non-empty 'phases' list: {resolved}")

    common = curriculum.get("common", {})
    if common is None:
        common = {}
    if not isinstance(common, dict):
        raise TypeError(f"Curriculum 'common' must be a JSON object: {resolved}")

    normalized_phases = []
    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            raise TypeError(f"Curriculum phase #{index} must be a JSON object: {resolved}")
        normalized_phases.append(phase)

    curriculum["common"] = common
    curriculum["phases"] = normalized_phases
    curriculum["_resolved_path"] = resolved
    curriculum["_base_dir"] = os.path.dirname(resolved)

    return curriculum


def resolve_clip_reward(args, hyperparams: Optional[Dict[str, Any]]) -> bool:
    if hasattr(args, "clip_reward") and getattr(args, "clip_reward") is not None:
        return bool(getattr(args, "clip_reward"))

    if hyperparams and "clip_reward" in hyperparams:
        return bool(hyperparams.get("clip_reward"))

    return True


def resolve_sticky_action_settings(
    requested_enabled: bool,
    hyperparams: Optional[Dict[str, Any]],
    *,
    default_prob: float = 0.25,
) -> tuple[bool, float]:
    if not requested_enabled:
        return False, -1.0

    enabled = True
    probability = default_prob

    if hyperparams and "sticky_actions" in hyperparams:
        enabled = bool(hyperparams.get("sticky_actions"))

    if hyperparams and "sticky_action_prob" in hyperparams:
        probability = float(hyperparams.get("sticky_action_prob"))

    if not enabled:
        return False, -1.0

    probability = max(0.0, min(1.0, probability))
    return True, probability
