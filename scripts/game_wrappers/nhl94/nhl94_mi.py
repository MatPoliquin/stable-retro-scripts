"""NHL94 model input helpers.

The wrapper uses fixed roster slots with absolute and puck-relative entity
features. Control and possession are represented as per-entity flags.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from game_wrappers.nhl94.nhl94_const import GameConsts


FeatureList = List[float]
FeatureVector = Tuple[float, ...]
ModelInputConfig = Optional[Dict[str, Any]]
FieldGetter = Callable[[Any, Any], float]

GOALIE_STAT_FIELDS = [
    "glove_left",
    "glove_right",
    "stick_left",
    "stick_right",
    "puck_control",
    "agility",
    "speed",
    "passing",
    "endurance",
    "weight",
]

BASE_SKATER_FIELDS = [
    "x",
    "y",
    "vx",
    "vy",
    "rel_puck_x",
    "rel_puck_y",
    "rel_puck_vx",
    "rel_puck_vy",
    "dist_to_puck",
    "dist_to_controlled",
    "ori_x",
    "ori_y",
    "has_puck",
    "passing_lane_clear",
    "one_timer_lane_good",
    "clear_shot_lane",
    "open_net_shot",
    "is_one_timer",
    "is_breakaway",
    "is_falling",
    "anim_frame",
    "has_anim",
]

FRIENDLY_SKATER_FIELDS = [
    *BASE_SKATER_FIELDS[:11],
    "is_controlled",
    *BASE_SKATER_FIELDS[11:],
]
OPPONENT_SKATER_FIELDS = BASE_SKATER_FIELDS

BASE_GOALIE_FIELDS = [
    "x",
    "y",
    "vx",
    "vy",
    "rel_puck_x",
    "rel_puck_y",
    "rel_puck_vx",
    "rel_puck_vy",
    "dist_to_puck",
    "dist_to_controlled",
    "has_puck",
    "is_pad_stack",
    "is_dive",
    "anim_frame",
    "has_anim",
]

FRIENDLY_GOALIE_FIELDS = [
    *BASE_GOALIE_FIELDS[:9],
    "is_controlled",
    *BASE_GOALIE_FIELDS[9:],
]
OPPONENT_GOALIE_FIELDS = BASE_GOALIE_FIELDS

ENGINE_FIELDS = [
    "engine.shot_mode_active",
    "engine.shot_taken",
    "engine.in_close_top_shelf",
    "engine.one_timer_collision_mode",
    "engine.breakaway_context",
    "engine.controlled_is_shooter",
    "engine.goalie_box_small",
    "engine.pass_dir",
    "engine.pass_speed",
]

DEFAULT_MODEL_INPUT_GROUPS = {
    "friendly_player": FRIENDLY_SKATER_FIELDS,
    "friendly_goalie": FRIENDLY_GOALIE_FIELDS,
    "opponent_player": OPPONENT_SKATER_FIELDS,
    "opponent_goalie": OPPONENT_GOALIE_FIELDS,
    "puck": [
        "x",
        "y",
        "vx",
        "vy",
    ],
    "goalie_stats": [
        *(f"team1_{field_name}" for field_name in GOALIE_STAT_FIELDS),
        *(f"team2_{field_name}" for field_name in GOALIE_STAT_FIELDS),
    ],
    "possession": [
        "team1_player_haspuck",
        "team1_goalie_haspuck",
        "team2_player_haspuck",
        "team2_goalie_haspuck",
        "team1_has_puck",
        "team2_has_puck",
        "puck_loose",
    ],
    "net": ["y", "left", "right", "depth", "rel_puck_y", "rel_puck_left", "rel_puck_right"],
    "buttons": ["up", "down", "left", "right", "b", "c", "slapshot_frames_held"],
    "hidden_state": ENGINE_FIELDS,
}


def _attr(name: str) -> FieldGetter:
    return lambda game_state, source: getattr(source, name)


SKATER_FIELD_GETTERS: Dict[str, FieldGetter] = {
    field_name: _attr(field_name) for field_name in FRIENDLY_SKATER_FIELDS
}
GOALIE_FIELD_GETTERS: Dict[str, FieldGetter] = {
    field_name: _attr(field_name) for field_name in FRIENDLY_GOALIE_FIELDS
}
PUCK_FIELD_GETTERS: Dict[str, FieldGetter] = {
    "x": lambda game_state, source: game_state.nz_puck.x,
    "y": lambda game_state, source: game_state.nz_puck.y,
    "vx": lambda game_state, source: game_state.nz_puck.vx,
    "vy": lambda game_state, source: game_state.nz_puck.vy,
}


def _team_entity_has_puck(team) -> bool:
    return bool(team.goalie.has_puck or any(player.has_puck for player in team.players))


GOALIE_STATS_FIELD_GETTERS: Dict[str, FieldGetter] = {
    **{
        f"team1_{field_name}": (lambda game_state, source, name=field_name: getattr(game_state.team1.nz_goalie_stats, name))
        for field_name in GOALIE_STAT_FIELDS
    },
    **{
        f"team2_{field_name}": (lambda game_state, source, name=field_name: getattr(game_state.team2.nz_goalie_stats, name))
        for field_name in GOALIE_STAT_FIELDS
    },
}
POSSESSION_FIELD_GETTERS: Dict[str, FieldGetter] = {
    "team1_player_haspuck": lambda game_state, source: game_state.team1.nz_player_haspuck,
    "team1_goalie_haspuck": lambda game_state, source: game_state.team1.nz_goalie_haspuck,
    "team2_player_haspuck": lambda game_state, source: game_state.team2.nz_player_haspuck,
    "team2_goalie_haspuck": lambda game_state, source: game_state.team2.nz_goalie_haspuck,
    "team1_has_puck": lambda game_state, source: float(_team_entity_has_puck(game_state.team1)),
    "team2_has_puck": lambda game_state, source: float(_team_entity_has_puck(game_state.team2)),
    "puck_loose": lambda game_state, source: float(
        not _team_entity_has_puck(game_state.team1)
        and not _team_entity_has_puck(game_state.team2)
    ),
}
NET_FIELD_GETTERS: Dict[str, FieldGetter] = {
    "y": lambda game_state, source: game_state.team2.nz_net.y,
    "left": lambda game_state, source: game_state.team2.nz_net.left,
    "right": lambda game_state, source: game_state.team2.nz_net.right,
    "depth": lambda game_state, source: game_state.team2.nz_net.depth,
    "rel_puck_y": lambda game_state, source: (game_state.team2.net.y - game_state.puck.y) / GameConsts.MAX_PUCK_Y,
    "rel_puck_left": lambda game_state, source: (game_state.team2.net.left - game_state.puck.x) / GameConsts.MAX_PUCK_X,
    "rel_puck_right": lambda game_state, source: (game_state.team2.net.right - game_state.puck.x) / GameConsts.MAX_PUCK_X,
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
    "friendly_player": SKATER_FIELD_GETTERS,
    "friendly_goalie": GOALIE_FIELD_GETTERS,
    "opponent_player": SKATER_FIELD_GETTERS,
    "opponent_goalie": GOALIE_FIELD_GETTERS,
    "puck": PUCK_FIELD_GETTERS,
    "goalie_stats": GOALIE_STATS_FIELD_GETTERS,
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
    return (
        (num_players * len(groups["friendly_player"]))
        + len(groups["friendly_goalie"])
        + (num_players * len(groups["opponent_player"]))
        + len(groups["opponent_goalie"])
        + len(groups["puck"])
        + len(groups["goalie_stats"])
        + len(groups["possession"])
        + len(groups["net"])
        + len(groups["buttons"])
        + len(groups["hidden_state"])
    )


def _base_features(game_state, model_input_config: ModelInputConfig) -> FeatureList:
    groups = _normalize_model_input_config(model_input_config)
    t1 = game_state.team1
    t2 = game_state.team2
    features: FeatureList = []

    for index in range(game_state.numPlayers):
        _append_fields(features, game_state, t1.nz_players[index], "friendly_player", groups["friendly_player"])

    _append_fields(features, game_state, t1.nz_goalie, "friendly_goalie", groups["friendly_goalie"])

    for index in range(game_state.numPlayers):
        _append_fields(features, game_state, t2.nz_players[index], "opponent_player", groups["opponent_player"])

    _append_fields(features, game_state, t2.nz_goalie, "opponent_goalie", groups["opponent_goalie"])
    _append_fields(features, game_state, None, "puck", groups["puck"])
    _append_fields(features, game_state, None, "goalie_stats", groups["goalie_stats"])
    _append_fields(features, game_state, None, "possession", groups["possession"])
    _append_fields(features, game_state, t2.nz_net, "net", groups["net"])
    return features


def _button_features(game_state, model_input_config: ModelInputConfig) -> FeatureList:
    groups = _normalize_model_input_config(model_input_config)
    features: FeatureList = []
    _append_fields(features, game_state, None, "buttons", groups["buttons"])
    return features


def _all_hidden_state_features(game_state) -> Dict[str, float]:
    engine = game_state.engine
    return {
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
