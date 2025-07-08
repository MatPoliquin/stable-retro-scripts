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

def init_model_5p(num_players):
    # Original 48 + 2 orientation components per player (5 controlled + 5 opponents)
    return 48 + (10 * 2)  # 68 total

def set_model_input_5p(game_state):
    t1 = game_state.team1
    t2 = game_state.team2

    # Get all player positions, velocities and orientations
    p1_players = [(t1.nz_players[i].x, t1.nz_players[i].y,
                   t1.nz_players[i].vx, t1.nz_players[i].vy,
                   t1.nz_players[i].ori_x, t1.nz_players[i].ori_y) for i in range(5)]

    # Determine controlled player (first slot should be the puck controller)
    controlled_idx = max(0, t1.control - 1) if t1.control > 0 else 0

    # Reorder so controlled player is first
    if controlled_idx != 0:
        p1_players[0], p1_players[controlled_idx] = p1_players[controlled_idx], p1_players[0]

    # Team 2 players (maintain original order)
    t2_players = [(t2.nz_players[i].x, t2.nz_players[i].y,
                   t2.nz_players[i].vx, t2.nz_players[i].vy,
                   t2.nz_players[i].ori_x, t2.nz_players[i].ori_y) for i in range(5)]

    # Flatten all player data
    t1_data = []
    for player_data in p1_players:
        t1_data.extend(player_data)

    t2_data = []
    for player_data in t2_players:
        t2_data.extend(player_data)

    return tuple(t1_data + t2_data + [
        game_state.nz_puck.x, game_state.nz_puck.y,
        game_state.nz_puck.vx, game_state.nz_puck.vy,
        t2.nz_goalie.x, t2.nz_goalie.y,
        t1.nz_player_haspuck, t2.nz_goalie_haspuck
    ])

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
