"""
NHL94 Model Input function
"""

import random
from typing import Tuple, Callable
from game_wrappers.nhl94_const import GameConsts



def init_model_1p(num_players):
    # Original 16 + 2 orientation components for each player (1 controlled + 1 opponent)
    return 16 + 2 + 2  # 20 total

def set_model_input_1p(game_state):
    t1 = game_state.team1
    t2 = game_state.team2

    return (t1.nz_players[0].x, t1.nz_players[0].y, \
            t1.nz_players[0].vx, t1.nz_players[0].vy, \
            t1.nz_players[0].ori_x, t1.nz_players[0].ori_y, \
            t2.nz_players[0].x, t2.nz_players[0].y, \
            t2.nz_players[0].vx, t2.nz_players[0].vy, \
            t2.nz_players[0].ori_x, t2.nz_players[0].ori_y, \
            game_state.nz_puck.x, game_state.nz_puck.y, \
            game_state.nz_puck.vx, game_state.nz_puck.vy, \
            t2.nz_goalie.x, t2.nz_goalie.y, \
            t1.nz_player_haspuck, t2.nz_goalie_haspuck)

def init_model_2p(num_players):
    # Original 24 + 2 orientation components per player (2 controlled + 2 opponents)
    return 24 + (4 * 2)  # 32 total

def set_model_input_2p(game_state):
    t1 = game_state.team1
    t2 = game_state.team2

    p1_x, p1_y = t1.nz_players[0].x, t1.nz_players[0].y
    p1_vel_x, p1_vel_y = t1.nz_players[0].vx, t1.nz_players[0].vy
    p1_ori_x, p1_ori_y = t1.nz_players[0].ori_x, t1.nz_players[0].ori_y
    p1_2_x, p1_2_y = t1.nz_players[1].x, t1.nz_players[1].y
    p1_2_vel_x, p1_2_vel_y = t1.nz_players[1].vx, t1.nz_players[1].vy
    p1_2_ori_x, p1_2_ori_y = t1.nz_players[1].ori_x, t1.nz_players[1].ori_y

    # Swap controlled player to first position if needed
    if t1.control == 2:
        p1_x, p1_2_x = p1_2_x, p1_x
        p1_y, p1_2_y = p1_2_y, p1_y
        p1_vel_x, p1_2_vel_x = p1_2_vel_x, p1_vel_x
        p1_vel_y, p1_2_vel_y = p1_2_vel_y, p1_vel_y
        p1_ori_x, p1_2_ori_x = p1_2_ori_x, p1_ori_x
        p1_ori_y, p1_2_ori_y = p1_2_ori_y, p1_ori_y

    return (p1_x, p1_y, \
            p1_vel_x, p1_vel_y, \
            p1_ori_x, p1_ori_y, \
            p1_2_x, p1_2_y, \
            p1_2_vel_x, p1_2_vel_y, \
            p1_2_ori_x, p1_2_ori_y, \
            t2.nz_players[0].x, t2.nz_players[0].y, \
            t2.nz_players[0].vx, t2.nz_players[0].vy, \
            t2.nz_players[0].ori_x, t2.nz_players[0].ori_y, \
            t2.nz_players[1].x, t2.nz_players[1].y, \
            t2.nz_players[1].vx, t2.nz_players[1].vy, \
            t2.nz_players[1].ori_x, t2.nz_players[1].ori_y, \
            game_state.nz_puck.x, game_state.nz_puck.y, \
            game_state.nz_puck.vx, game_state.nz_puck.vy, \
            t2.nz_goalie.x, t2.nz_goalie.y, \
            t1.nz_player_haspuck, t2.nz_goalie_haspuck)


def init_model(num_players: int) -> int:
    """Initialize model input size based on number of players per team.

    Args:
        num_players: Number of players per team (1, 2, or 5)

    Returns:
        Total size of the model input vector
    """
    # Each player has 6 features (x, y, vx, vy, ori_x, ori_y)
    # Puck has 4 features (x, y, vx, vy)
    # Opponent goalie has 2 features (x, y)
    # Plus 2 binary features for puck possession
    return (num_players * 6 * 2) + 4 + 2 + 2

def set_model_input(game_state) -> Tuple[float, ...]:
    """Create unified model input vector for any number of players.

    Args:
        game_state: The current game state object

    Returns:
        Tuple of normalized values for model input
    """
    t1 = game_state.team1
    t2 = game_state.team2
    num_players = game_state.numPlayers

    # Collect all player data - controlled team first
    t1_players = []
    t2_players = []

    # Reorder controlled team so controlled player is first
    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0
    player_order = list(range(num_players))
    if controlled_idx != 0:
        player_order[0], player_order[controlled_idx] = player_order[controlled_idx], player_order[0]

    # Add controlled team players in order
    for i in player_order:
        player = t1.nz_players[i]
        t1_players.extend([
            player.x, player.y,
            player.vx, player.vy,
            player.ori_x, player.ori_y
        ])

    # Add opponent team players in original order
    for i in range(num_players):
        player = t2.nz_players[i]
        t2_players.extend([
            player.x, player.y,
            player.vx, player.vy,
            player.ori_x, player.ori_y
        ])

    # Combine all features
    return tuple(t1_players + t2_players + [
        game_state.nz_puck.x, game_state.nz_puck.y,
        game_state.nz_puck.vx, game_state.nz_puck.vy,
        t2.nz_goalie.x, t2.nz_goalie.y,
        t1.nz_player_haspuck, t2.nz_goalie_haspuck
    ])

#==================================
# Uses relative puck pos and remove puck vel
#==================================
def init_model_rel(num_players: int) -> int:
    """Initialize model input size based on number of players per team.

    Args:
        num_players: Number of players per team (1, 2, or 5)

    Returns:
        Total size of the model input vector
    """
    # Each player has 6 features (x, y, vx, vy, ori_x, ori_y)
    # Puck has 2 features (x, y) - position only
    # Opponent goalie has 2 features (x, y)
    # Plus 2 binary features for puck possession
    # Plus 2 features for puck relative to controlled player
    return (num_players * 6 * 2) + 2 + 2 + 2 + 2

def set_model_input_rel(game_state) -> Tuple[float, ...]:
    """Create unified model input vector for any number of players.

    Args:
        game_state: The current game state object

    Returns:
        Tuple of normalized values for model input
    """
    t1 = game_state.team1
    t2 = game_state.team2
    num_players = game_state.numPlayers

    # Collect all player data - controlled team first
    t1_players = []
    t2_players = []

    # Reorder controlled team so controlled player is first
    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0
    player_order = list(range(num_players))
    if controlled_idx != 0:
        player_order[0], player_order[controlled_idx] = player_order[controlled_idx], player_order[0]

    # Calculate puck position relative to controlled player
    controlled_player = t1.players[controlled_idx] if t1.control > 0 else t1.players[0]
    puck_rel_controlled_x = game_state.puck.x - controlled_player.x
    puck_rel_controlled_y = game_state.puck.y - controlled_player.y

    # Normalize relative puck position
    nz_puck_rel_controlled_x = puck_rel_controlled_x / 30
    nz_puck_rel_controlled_y = puck_rel_controlled_y / 30

    # Add controlled team players in order (without relative puck position)
    for i in player_order:
        player = t1.nz_players[i]
        if i == 0:
            t1_players.extend([
                player.x, player.y,
                player.vx, player.vy,
                player.ori_x, player.ori_y  # Removed rel_puck_x/y
            ])
        else:
            t1_players.extend([
                player.rel_controlled_x, player.rel_controlled_y,
                player.rel_controlled_vx, player.rel_controlled_vy,
                player.ori_x, player.ori_y  # Removed rel_puck_x/y
            ])

    # Add opponent team players in original order (without relative puck position)
    for i in range(num_players):
        player = t2.nz_players[i]
        t2_players.extend([
            player.rel_controlled_x, player.rel_controlled_y,
            player.rel_controlled_vx, player.rel_controlled_vy,
            player.ori_x, player.ori_y  # Removed rel_puck_x/y
        ])

    # Combine all features
    return tuple(t1_players + t2_players + [
        game_state.nz_puck.x, game_state.nz_puck.y,  # Absolute puck position
        t2.nz_goalie.x, t2.nz_goalie.y,
        t1.nz_player_haspuck, t2.nz_goalie_haspuck,
        nz_puck_rel_controlled_x, nz_puck_rel_controlled_y  # Puck relative to controlled player
    ])

def init_model_rel_dist(num_players: int) -> int:
    """Initialize model input size with relative positions and distances to controlled player.

    Args:
        num_players: Number of players per team (1, 2, or 5)

    Returns:
        Total size of the model input vector
    """
    # Features per controlled player: 6 (x,y,vx,vy,ori_x,ori_y)
    # Features per teammate: 6 (rel_x,rel_y,rel_vx,rel_vy,ori_x,ori_y) + 1 distance + 1 passable
    # Features per opponent: 6 (rel_x,rel_y,rel_vx,rel_vy,ori_x,ori_y) + 1 distance
    # Puck features: 2 (x,y) + 2 (rel_x,rel_y) + dist
    # Goalie features: 2 (x,y)
    # Possession flags: 2
    # Net features: 6 per net (absolute y, left, right + relative y, left, right) * 1 net = 6
    return (6 + ((num_players-1)*7) + (num_players*7) + 4 + 2 + 2 + 1 + (num_players - 1)) + 6

def set_model_input_rel_dist(game_state) -> Tuple[float, ...]:
    """Standalone version with relative positions and distances to controlled player.

    Args:
        game_state: The current game state object

    Returns:
        Tuple of normalized values for model input
    """
    t1 = game_state.team1
    t2 = game_state.team2
    num_players = game_state.numPlayers

    # Find controlled player (default to first player if goalie is controlled)
    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0
    controlled_player = t1.nz_players[controlled_idx]

    # Initialize feature list
    features = []

    # 1. Add controlled player's absolute position and orientation
    features.extend([
        controlled_player.x,
        controlled_player.y,
        controlled_player.vx,
        controlled_player.vy,
        controlled_player.ori_x,
        controlled_player.ori_y
    ])

    # 2. Add teammates' relative positions and distances (skip controlled player)
    for i in range(num_players):
        if i != controlled_idx:
            player = t1.nz_players[i]
            features.extend([
                player.rel_controlled_x,
                player.rel_controlled_y,
                player.rel_controlled_vx,
                player.rel_controlled_vy,
                player.ori_x,
                player.ori_y,
                player.dist_to_controlled,
                player.passing_lane_clear
            ])

    # 3. Add opponents' relative positions and distances
    for i in range(num_players):
        player = t2.nz_players[i]
        features.extend([
            player.rel_controlled_x,
            player.rel_controlled_y,
            player.rel_controlled_vx,
            player.rel_controlled_vy,
            player.ori_x,
            player.ori_y,
            player.dist_to_controlled_opp  # Already normalized
        ])

    # 4. Add puck information (absolute and relative to controlled)
    features.extend([
        game_state.nz_puck.x,
        game_state.nz_puck.y,
        controlled_player.rel_puck_x,  # Relative to controlled player
        controlled_player.rel_puck_y,   # Relative to controlled player
        controlled_player.dist_to_puck  # Relative to controlled player
    ])

    # 5. Add opponent goalie position
    features.extend([
        t2.nz_goalie.x,
        t2.nz_goalie.y
    ])

    # 6. Add possession flags
    features.extend([
        t1.nz_player_haspuck,
        t2.nz_goalie_haspuck
    ])

    # 7. Add net coordinates (both absolute and relative to controlled player)
    # Team 1 net (our net)
    # features.extend([
    #     t1.nz_net.y,                   # Absolute Y position
    #     t1.nz_net.left,                 # Absolute left post
    #     t1.nz_net.right,                # Absolute right post
    #     t1.nz_net.rel_controlled_y,     # Relative Y to controlled player
    #     t1.nz_net.rel_controlled_left,  # Relative left post
    #     t1.nz_net.rel_controlled_right  # Relative right post
    # ])

    # Team 2 net (opponent net)
    features.extend([
        t2.nz_net.y,                   # Absolute Y position
        t2.nz_net.left,                 # Absolute left post
        t2.nz_net.right,                # Absolute right post
        t2.nz_net.rel_controlled_y,     # Relative Y to controlled player
        t2.nz_net.rel_controlled_left,  # Relative left post
        t2.nz_net.rel_controlled_right  # Relative right post
    ])

    return tuple(features)

def init_model_rel_dist_buttons(num_players: int) -> int:
    """Initialize model input size with relative positions, distances, and button states.

    Args:
        num_players: Number of players per team (1, 2, or 5)

    Returns:
        Total size of the model input vector
    """
    # Same as rel_dist model but adds:
    # - Current button states (6 buttons)
    # - Slapshot frames held (1 value)
    return init_model_rel_dist(num_players) + 6 + 1

def set_model_input_rel_dist_buttons(game_state) -> Tuple[float, ...]:
    """Model input with relative positions, distances, and button states for slapshots.

    Args:
        game_state: The current game state object

    Returns:
        Tuple of normalized values for model input including button states
    """
    # Get all features from the rel_dist version
    features = list(set_model_input_rel_dist(game_state))

    # Add button states (normalized to 0/1)
    button_features = [
        float(game_state.action[0]),  # Up
        float(game_state.action[1]),  # Down
        float(game_state.action[2]),  # Left
        float(game_state.action[3]),  # Right
        float(game_state.action[4]),  # B button
        float(game_state.action[5]),  # C button
    ]

    # Add slapshot frames held (normalized 0-1)
    slapshot_frames = getattr(game_state, 'slapshot_frames_held', 0)
    normalized_slapshot = min(slapshot_frames / 60.0, 1.0)  # Normalize to 0-1 based on max hold frames

    # Combine all features
    return tuple(features + button_features + [normalized_slapshot])