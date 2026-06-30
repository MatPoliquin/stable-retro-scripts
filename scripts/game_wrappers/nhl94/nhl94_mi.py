"""NHL94 model input helpers.

The current wrapper uses a single observation layout: relative skater features,
current button state, and decoded shot / goalie state.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


FeatureList = List[float]
FeatureVector = Tuple[float, ...]
ModelInputConfig = Optional[Dict[str, Any]]
FieldGetter = Callable[[Any, Any], float]

DEFAULT_MODEL_INPUT_GROUPS = {
    "controlled_player": ["x", "y", "vx", "vy", "ori_x", "ori_y", "clear_shot_lane", "open_net_shot"],
    "teammate": [
        "rel_controlled_x",
        "rel_controlled_y",
        "rel_controlled_vx",
        "rel_controlled_vy",
        "ori_x",
        "ori_y",
        "dist_to_controlled",
        "passing_lane_clear",
        "clear_shot_lane",
        "open_net_shot",
    ],
    "opponent": [
        "rel_controlled_x",
        "rel_controlled_y",
        "rel_controlled_vx",
        "rel_controlled_vy",
        "ori_x",
        "ori_y",
        "dist_to_controlled_opp",
        "clear_shot_lane",
        "open_net_shot",
    ],
    "puck": [
        "x",
        "y",
        "rel_controlled_x",
        "rel_controlled_y",
        "rel_controlled_vx",
        "rel_controlled_vy",
        "dist_to_controlled",
    ],
    "goalie": ["x", "y"],
    "possession": ["team1_player_haspuck", "team2_goalie_haspuck"],
    "net": ["y", "left", "right", "rel_controlled_y", "rel_controlled_left", "rel_controlled_right"],
    "buttons": ["up", "down", "left", "right", "b", "c", "slapshot_frames_held"],
    "hidden_state": [
        "controlled_player.is_one_timer",
        "controlled_player.is_breakaway",
        "controlled_player.is_falling",
        "controlled_player.anim_frame",
        "controlled_player.has_anim",
        "teammate.is_one_timer",
        "teammate.one_timer_lane_good",
        "opponent_goalie.is_pad_stack",
        "opponent_goalie.is_dive",
        "opponent_goalie.anim_frame",
        "engine.shot_mode_active",
        "engine.shot_taken",
        "engine.in_close_top_shelf",
        "engine.one_timer_collision_mode",
        "engine.breakaway_context",
        "engine.controlled_is_shooter",
        "engine.goalie_box_small",
        "engine.pass_dir",
        "engine.pass_speed",
    ],
}


def _attr(name: str) -> FieldGetter:
    return lambda game_state, source: getattr(source, name)


CONTROLLED_PLAYER_FIELD_GETTERS: Dict[str, FieldGetter] = {
    field_name: _attr(field_name) for field_name in DEFAULT_MODEL_INPUT_GROUPS["controlled_player"]
}
TEAMMATE_FIELD_GETTERS: Dict[str, FieldGetter] = {
    field_name: _attr(field_name) for field_name in DEFAULT_MODEL_INPUT_GROUPS["teammate"]
}
OPPONENT_FIELD_GETTERS: Dict[str, FieldGetter] = {
    field_name: _attr(field_name) for field_name in DEFAULT_MODEL_INPUT_GROUPS["opponent"]
}
PUCK_FIELD_GETTERS: Dict[str, FieldGetter] = {
    "x": lambda game_state, source: game_state.nz_puck.x,
    "y": lambda game_state, source: game_state.nz_puck.y,
    "rel_controlled_x": lambda game_state, source: source.rel_puck_x,
    "rel_controlled_y": lambda game_state, source: source.rel_puck_y,
    "rel_controlled_vx": lambda game_state, source: source.rel_puck_vx,
    "rel_controlled_vy": lambda game_state, source: source.rel_puck_vy,
    "dist_to_controlled": lambda game_state, source: source.dist_to_puck,
}
GOALIE_FIELD_GETTERS: Dict[str, FieldGetter] = {
    "x": lambda game_state, source: game_state.team2.nz_goalie.x,
    "y": lambda game_state, source: game_state.team2.nz_goalie.y,
}
POSSESSION_FIELD_GETTERS: Dict[str, FieldGetter] = {
    "team1_player_haspuck": lambda game_state, source: game_state.team1.nz_player_haspuck,
    "team2_goalie_haspuck": lambda game_state, source: game_state.team2.nz_goalie_haspuck,
}
NET_FIELD_GETTERS: Dict[str, FieldGetter] = {
    "y": lambda game_state, source: game_state.team2.nz_net.y,
    "left": lambda game_state, source: game_state.team2.nz_net.left,
    "right": lambda game_state, source: game_state.team2.nz_net.right,
    "rel_controlled_y": lambda game_state, source: game_state.team2.nz_net.rel_controlled_y,
    "rel_controlled_left": lambda game_state, source: game_state.team2.nz_net.rel_controlled_left,
    "rel_controlled_right": lambda game_state, source: game_state.team2.nz_net.rel_controlled_right,
}
BUTTON_FIELD_GETTERS: Dict[str, FieldGetter] = {
    "up": lambda game_state, source: float(game_state.action[0]),
    "down": lambda game_state, source: float(game_state.action[1]),
    "left": lambda game_state, source: float(game_state.action[2]),
    "right": lambda game_state, source: float(game_state.action[3]),
    "b": lambda game_state, source: float(game_state.action[4]),
    "c": lambda game_state, source: float(game_state.action[5]),
    "slapshot_frames_held": lambda game_state, source: min(getattr(game_state, "slapshot_frames_held", 0) / 60.0, 1.0),
}


MODEL_INPUT_FIELD_GETTERS: Dict[str, Dict[str, FieldGetter]] = {
    "controlled_player": CONTROLLED_PLAYER_FIELD_GETTERS,
    "teammate": TEAMMATE_FIELD_GETTERS,
    "opponent": OPPONENT_FIELD_GETTERS,
    "puck": PUCK_FIELD_GETTERS,
    "goalie": GOALIE_FIELD_GETTERS,
    "possession": POSSESSION_FIELD_GETTERS,
    "net": NET_FIELD_GETTERS,
    "buttons": BUTTON_FIELD_GETTERS,
}


def _normalize_model_input_config(model_input_config: ModelInputConfig) -> Dict[str, List[str]]:
    if model_input_config is None:
        return {group: list(fields) for group, fields in DEFAULT_MODEL_INPUT_GROUPS.items()}
    if not isinstance(model_input_config, dict):
        raise TypeError("model_input must be a JSON object when provided")

    groups_config = model_input_config.get("groups", model_input_config)
    if not isinstance(groups_config, dict):
        raise TypeError("model_input['groups'] must be a JSON object when provided")

    strict = bool(model_input_config.get("strict", True))
    normalized = {group: list(fields) for group, fields in DEFAULT_MODEL_INPUT_GROUPS.items()}

    for group_name, fields in groups_config.items():
        if group_name in ("schema_version", "strict", "groups"):
            continue
        if group_name not in DEFAULT_MODEL_INPUT_GROUPS:
            if strict:
                raise ValueError(f"Unsupported NHL94 model_input group: {group_name}")
            continue
        if fields is None:
            normalized[group_name] = []
            continue
        if isinstance(fields, bool):
            normalized[group_name] = list(DEFAULT_MODEL_INPUT_GROUPS[group_name]) if fields else []
            continue
        if not isinstance(fields, list):
            raise TypeError(f"model_input group '{group_name}' must be a list, bool, or null")

        known_fields = set(DEFAULT_MODEL_INPUT_GROUPS[group_name])
        selected_fields = []
        for field_name in fields:
            if not isinstance(field_name, str):
                raise TypeError(f"model_input group '{group_name}' field names must be strings")
            if field_name not in known_fields:
                if strict:
                    raise ValueError(f"Unsupported NHL94 model_input field: {group_name}.{field_name}")
                continue
            selected_fields.append(field_name)
        normalized[group_name] = selected_fields

    return normalized


def _append_fields(
    features: FeatureList,
    game_state,
    source,
    group_name: str,
    field_names: Iterable[str],
) -> None:
    getters = MODEL_INPUT_FIELD_GETTERS[group_name]
    for field_name in field_names:
        features.append(getters[field_name](game_state, source))


def init_model(num_players: int, model_input_config: ModelInputConfig = None) -> int:
    """Return the size of the canonical scalar observation vector."""
    groups = _normalize_model_input_config(model_input_config)
    teammate_count = max(num_players - 1, 0)
    return (
        len(groups["controlled_player"])
        + (teammate_count * len(groups["teammate"]))
        + (num_players * len(groups["opponent"]))
        + len(groups["puck"])
        + len(groups["goalie"])
        + len(groups["possession"])
        + len(groups["net"])
        + len(groups["buttons"])
        + len(groups["hidden_state"])
    )


def _controlled_player_index(team) -> int:
    return max(0, team.control - 1) if team.control > 0 else 0


def _base_features(game_state, model_input_config: ModelInputConfig) -> FeatureList:
    groups = _normalize_model_input_config(model_input_config)
    t1 = game_state.team1
    t2 = game_state.team2
    controlled_idx = _controlled_player_index(t1)
    controlled_player = t1.nz_players[controlled_idx]
    features: FeatureList = []

    _append_fields(features, game_state, controlled_player, "controlled_player", groups["controlled_player"])

    for index in range(game_state.numPlayers):
        if index == controlled_idx:
            continue
        _append_fields(features, game_state, t1.nz_players[index], "teammate", groups["teammate"])

    for index in range(game_state.numPlayers):
        _append_fields(features, game_state, t2.nz_players[index], "opponent", groups["opponent"])

    _append_fields(features, game_state, controlled_player, "puck", groups["puck"])
    _append_fields(features, game_state, t2.nz_goalie, "goalie", groups["goalie"])
    _append_fields(features, game_state, None, "possession", groups["possession"])
    _append_fields(features, game_state, t2.nz_net, "net", groups["net"])
    return features


def _button_features(game_state, model_input_config: ModelInputConfig) -> FeatureList:
    groups = _normalize_model_input_config(model_input_config)
    features: FeatureList = []
    _append_fields(features, game_state, None, "buttons", groups["buttons"])
    return features


def _all_hidden_state_features(game_state) -> Dict[str, float]:
    t1 = game_state.team1
    t2 = game_state.team2
    engine = game_state.engine

    controlled_player = t1.get_controlled_player()
    opponent_goalie = t2.goalie
    controlled_idx = t1.control - 1 if t1.control > 0 else -1

    teammate_one_timer = 0.0
    teammate_one_timer_good = 0.0
    for index, player in enumerate(t1.players):
        if index == controlled_idx:
            continue
        teammate_one_timer = max(teammate_one_timer, player.is_one_timer)
        if player.one_timer_lane_good:
            teammate_one_timer_good = 1.0

    return {
        "controlled_player.is_one_timer": controlled_player.is_one_timer,
        "controlled_player.is_breakaway": controlled_player.is_breakaway,
        "controlled_player.is_falling": controlled_player.is_falling,
        "controlled_player.anim_frame": min(abs(controlled_player.anim_frame) / 32.0, 1.0),
        "controlled_player.has_anim": float(controlled_player.anim != 0),
        "teammate.is_one_timer": teammate_one_timer,
        "teammate.one_timer_lane_good": teammate_one_timer_good,
        "opponent_goalie.is_pad_stack": opponent_goalie.is_pad_stack,
        "opponent_goalie.is_dive": opponent_goalie.is_dive,
        "opponent_goalie.anim_frame": min(abs(opponent_goalie.anim_frame) / 32.0, 1.0),
        "engine.shot_mode_active": engine.shot_mode_active,
        "engine.shot_taken": engine.shot_taken,
        "engine.in_close_top_shelf": engine.in_close_top_shelf,
        "engine.one_timer_collision_mode": engine.one_timer_collision_mode,
        "engine.breakaway_context": engine.breakaway_context,
        "engine.controlled_is_shooter": engine.controlled_is_shooter,
        "engine.goalie_box_small": engine.goalie_box_small,
        "engine.pass_dir": min(engine.pass_dir / 7.0, 1.0),
        "engine.pass_speed": min(engine.pass_speed / 255.0, 1.0),
    }


def _hidden_state_features(game_state, model_input_config: ModelInputConfig) -> FeatureList:
    groups = _normalize_model_input_config(model_input_config)
    hidden_features = _all_hidden_state_features(game_state)
    return [hidden_features[field_name] for field_name in groups["hidden_state"]]


def set_model_input(game_state, model_input_config: ModelInputConfig = None) -> FeatureVector:
    """Build the canonical scalar observation vector."""
    features = _base_features(game_state, model_input_config)
    features.extend(_button_features(game_state, model_input_config))
    features.extend(_hidden_state_features(game_state, model_input_config))
    assert len(features) == init_model(game_state.numPlayers, model_input_config)
    return tuple(features)
