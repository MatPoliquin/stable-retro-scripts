"""
NHL94 Debug Display with Player Input
"""

import pygame
import numpy as np
from game_wrappers.nhl94_const import GameConsts
from pygame import gfxdraw
from game_wrappers.nhl94_gamestate import NHL94GameState

class NHL94DebugDisplay:
    DEBUG_WIDTH = 800
    DEBUG_HEIGHT = 800
    GAME_WIDTH = 320 * 2  # Half size of original display
    GAME_HEIGHT = 240 * 2

    # Colors
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (255, 0, 0)
    COLOR_BLUE = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (255, 255, 0)
    COLOR_CYAN = (0, 255, 255)
    COLOR_ICE = (200, 230, 255)
    COLOR_LINE = (150, 150, 150)
    COLOR_BLACK = (0, 0, 0)
    COLOR_PURPLE = (128, 0, 128)

    # Display parameters
    PLAYER_RADIUS = 10
    PUCK_RADIUS = 5
    VELOCITY_SCALE = 5
    ORIENTATION_LENGTH = 20

    def __init__(self, env, args, total_params, nn_type, button_names):
        self.env = env
        self.args = args

        self.game_state = env.get_attr("game_state")[0]

        pygame.init()
        self.screen = pygame.display.set_mode((self.DEBUG_WIDTH + self.GAME_WIDTH,
                                             max(self.DEBUG_HEIGHT, self.GAME_HEIGHT)))
        self.font = pygame.font.SysFont('Arial', 16)
        self.big_font = pygame.font.SysFont('Arial', 24)

        # Create surfaces
        self.debug_surf = pygame.Surface((self.DEBUG_WIDTH, self.DEBUG_HEIGHT))
        self.game_surf = pygame.Surface((self.GAME_WIDTH, self.GAME_HEIGHT))

        # Scale factors
        self.scale_x = self.DEBUG_WIDTH / (2 * GameConsts.MAX_PUCK_X)
        self.scale_y = self.DEBUG_HEIGHT / (2 * GameConsts.MAX_PUCK_Y)

        # Input state
        self.player_actions = [0] * GameConsts.INPUT_MAX
        self.key_action_map = {
            pygame.K_UP: GameConsts.INPUT_UP,
            pygame.K_DOWN: GameConsts.INPUT_DOWN,
            pygame.K_LEFT: GameConsts.INPUT_LEFT,
            pygame.K_RIGHT: GameConsts.INPUT_RIGHT,
            pygame.K_z: GameConsts.INPUT_A,
            pygame.K_x: GameConsts.INPUT_B,
            pygame.K_c: GameConsts.INPUT_C,
            pygame.K_a: GameConsts.INPUT_X,
            pygame.K_s: GameConsts.INPUT_Y,
            pygame.K_d: GameConsts.INPUT_Z,
            pygame.K_RETURN: GameConsts.INPUT_START,
            pygame.K_TAB: GameConsts.INPUT_MODE
        }

        # Control mode (AI or human)
        self.human_control = True
        self.control_help = [
            "CONTROLS:",
            "Arrows: Move",
            "Z: A Button (Pass)",
            "X: B Button (Slap Shot)",
            "C: C Button (Wrist Shot)",
            "A/S/D: X/Y/Z Buttons",
            "TAB: Mode Button",
            "ENTER: Start Button",
            "F1: Toggle AI/Human Control",
            "ESC: Quit",
            "1: Toggle Passing Lanes",
            "2: Toggle Velocities",
            "3: Toggle Orientations",
            "4: Toggle Distances",
            "F2: Save Screenshot"
        ]

        # Visualization toggles
        self.show_passing_lanes = True
        self.show_velocities = True
        self.show_orientations = True
        self.show_distances = True

    def close(self):
        pygame.quit()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def get_human_input(self):
        """Get input from keyboard and map to game actions"""
        keys = pygame.key.get_pressed()
        actions = [0] * GameConsts.INPUT_MAX

        for key, action in self.key_action_map.items():
            if keys[key]:
                actions[action] = 1

        return actions

    def step(self, action=None):
        """Step the environment with either AI or human input"""

        self.game_state = self.env.get_attr("game_state")[0]

        if self.human_control:
            action = self.get_human_input()

        # Convert to the format expected by the environment
        if action is None:
            env_action = [0] * 12  # Default no-op action
        elif isinstance(action, list) and len(action) == 12:
            # Already in correct format
            env_action = [int(a) for a in action]
        else:
            # Fallback to no-op
            env_action = [0] * 12

        obs, rew, done, info = self.env.step([env_action])

        # Draw the frame
        framebuffer = self.env.render()
        self.draw_frame(framebuffer, info, action)

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit("User requested exit")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    raise SystemExit("User requested exit")
                elif event.key == pygame.K_F1:
                    self.human_control = not self.human_control
                elif event.key == pygame.K_F2:
                    pygame.image.save(self.screen, "debug_screenshot.png")
                elif event.key == pygame.K_1:
                    self.show_passing_lanes = not self.show_passing_lanes
                elif event.key == pygame.K_2:
                    self.show_velocities = not self.show_velocities
                elif event.key == pygame.K_3:
                    self.show_orientations = not self.show_orientations
                elif event.key == pygame.K_4:
                    self.show_distances = not self.show_distances

        return obs, rew, done, info

    def _transform_coords(self, x, y):
        """Transform game coordinates to debug display coordinates with proper bounds checking"""
        tx = self.DEBUG_WIDTH/2 + x * self.scale_x
        ty = self.DEBUG_HEIGHT/2 - y * self.scale_y

        return int(tx), int(ty)

    def _draw_rink(self):
        """Draw the hockey rink with center lines and zones"""
        self.debug_surf.fill(self.COLOR_ICE)

        # Rink boundaries - use PLAYER bounds since they're larger than PUCK bounds
        rink_rect = pygame.Rect(
            self.DEBUG_WIDTH/2 - GameConsts.MAX_PUCK_X * self.scale_x,
            self.DEBUG_HEIGHT/2 - GameConsts.MAX_PUCK_Y * self.scale_y,
            2 * GameConsts.MAX_PUCK_X * self.scale_x,
            2 * GameConsts.MAX_PUCK_Y * self.scale_y
        )
        pygame.draw.rect(self.debug_surf, self.COLOR_LINE, rink_rect, 2)

        # Center line
        pygame.draw.line(
            self.debug_surf, self.COLOR_LINE,
            (self.DEBUG_WIDTH/2, rink_rect.top),
            (self.DEBUG_WIDTH/2, rink_rect.bottom),
            2
        )

        # Faceoff circles (simplified)
        pygame.draw.circle(
            self.debug_surf, self.COLOR_LINE,
            (int(self.DEBUG_WIDTH/2), int(self.DEBUG_HEIGHT/2)), 30, 1
        )

        # Draw nets at opposite ends
        # Team 1 (Red) net at bottom
        left_x, left_y = self._transform_coords(GameConsts.P1_NET_LEFT_POLL, GameConsts.P1_NET_Y)
        right_x, right_y = self._transform_coords(GameConsts.P1_NET_RIGHT_POLL, GameConsts.P1_NET_Y)
        back_y = GameConsts.P1_NET_Y + GameConsts.NET_DEPTH  # Note: + instead of - because Y increases downward
        back_left_x, back_left_y = self._transform_coords(GameConsts.P1_NET_LEFT_POLL, back_y)
        back_right_x, back_right_y = self._transform_coords(GameConsts.P1_NET_RIGHT_POLL, back_y)

        pygame.draw.polygon(
            self.debug_surf, self.COLOR_RED,
            [(left_x, left_y), (right_x, right_y),
            (back_right_x, back_right_y), (back_left_x, back_left_y)],
            2
        )

        # Team 2 (Blue) net at top
        left_x, left_y = self._transform_coords(GameConsts.P2_NET_LEFT_POLL, GameConsts.P2_NET_Y)
        right_x, right_y = self._transform_coords(GameConsts.P2_NET_RIGHT_POLL, GameConsts.P2_NET_Y)
        back_y = GameConsts.P2_NET_Y - GameConsts.NET_DEPTH  # Note: - because Y decreases upward
        back_left_x, back_left_y = self._transform_coords(GameConsts.P2_NET_LEFT_POLL, back_y)
        back_right_x, back_right_y = self._transform_coords(GameConsts.P2_NET_RIGHT_POLL, back_y)

        pygame.draw.polygon(
            self.debug_surf, self.COLOR_BLUE,
            [(left_x, left_y), (right_x, right_y),
            (back_right_x, back_right_y), (back_left_x, back_left_y)],
            2
        )

    def _draw_player(self, player, color, is_goalie=False):
        """Draw a player with position, orientation and velocity"""
        x, y = self._transform_coords(player.x, player.y)

        # Draw player circle
        radius = self.PLAYER_RADIUS + 5 if is_goalie else self.PLAYER_RADIUS
        pygame.draw.circle(self.debug_surf, color, (x, y), radius)

        # Draw orientation line if enabled
        if self.show_orientations:
            end_x = x + player.ori_x * self.ORIENTATION_LENGTH
            end_y = y + player.ori_y * self.ORIENTATION_LENGTH
            pygame.draw.line(self.debug_surf, self.COLOR_WHITE, (x, y), (end_x, end_y), 2)

        # Draw velocity vector if enabled
        if self.show_velocities:
            vel_x = x + player.vx * self.VELOCITY_SCALE
            vel_y = y + player.vy * self.VELOCITY_SCALE
            pygame.draw.line(self.debug_surf, self.COLOR_YELLOW, (x, y), (vel_x, vel_y), 2)

            # Draw small arrowhead for velocity
            pygame.draw.line(self.debug_surf, self.COLOR_YELLOW, (vel_x, vel_y),
                            (vel_x - 5, vel_y - 5), 2)
            pygame.draw.line(self.debug_surf, self.COLOR_YELLOW, (vel_x, vel_y),
                            (vel_x - 5, vel_y + 5), 2)

        return x, y

    def _draw_puck(self, puck):
        """Draw the puck with position and velocity"""
        x, y = self._transform_coords(puck.x, puck.y)

        # Draw puck
        pygame.draw.circle(self.debug_surf, self.COLOR_WHITE, (x, y), self.PUCK_RADIUS)

        # Draw velocity vector if enabled
        if self.show_velocities:
            vel_x = x + puck.vx * self.VELOCITY_SCALE
            vel_y = y + puck.vy * self.VELOCITY_SCALE
            pygame.draw.line(self.debug_surf, self.COLOR_RED, (x, y), (vel_x, vel_y), 2)

            # Draw small arrowhead for velocity
            pygame.draw.line(self.debug_surf, self.COLOR_RED, (vel_x, vel_y),
                            (vel_x - 3, vel_y - 3), 2)
            pygame.draw.line(self.debug_surf, self.COLOR_RED, (vel_x, vel_y),
                            (vel_x - 3, vel_y + 3), 2)

    def _draw_passing_lanes(self, team, color):
        """Draw passing lanes only for the player controlling the puck"""
        if not self.show_passing_lanes:
            return

        # Only draw if this team has puck control and it's a player (not goalie)
        if not (team.player_haspuck and team.control > 0):
            return

        controlled_player = team.players[team.control - 1]  # Get the controlled player

        # Draw line to each teammate
        for i, player in enumerate(team.players):
            if i != (team.control - 1):  # Skip the controlled player itself
                start_x, start_y = self._transform_coords(controlled_player.x, controlled_player.y)
                end_x, end_y = self._transform_coords(player.x, player.y)

                # Thicker passing lanes (3px width)
                lane_color = self.COLOR_GREEN if player.passing_lane_clear else self.COLOR_RED
                pygame.draw.line(self.debug_surf, lane_color, (start_x, start_y), (end_x, end_y), 3)

                # Larger midpoint indicator (5px radius)
                mid_x = (start_x + end_x) // 2
                mid_y = (start_y + end_y) // 2
                pygame.draw.circle(self.debug_surf, color, (mid_x, mid_y), 5)

    def _draw_controlled_distances(self, team, color):
        """Draw distances from controlled player to teammates"""
        if not self.show_distances:
            return

        if team.control == 0:  # Goalie is controlled
            controlled_player = team.goalie
        else:
            controlled_player = team.players[team.control - 1]

        # Draw distance to each teammate
        for i, player in enumerate(team.players):
            if i != (team.control - 1) or team.control == 0:  # Skip self if not goalie
                start_x, start_y = self._transform_coords(controlled_player.x, controlled_player.y)
                end_x, end_y = self._transform_coords(player.x, player.y)

                # Draw distance line
                pygame.draw.line(self.debug_surf, self.COLOR_PURPLE, (start_x, start_y), (end_x, end_y), 1)

                # Calculate midpoint for distance text
                mid_x = (start_x + end_x) // 2
                mid_y = (start_y + end_y) // 2

                # Display the distance value
                distance = GameConsts.Distance(
                    (controlled_player.x, controlled_player.y),
                    (player.x, player.y)
                )
                text = self.font.render(f"{distance:.1f}", True, self.COLOR_WHITE)
                self.debug_surf.blit(text, (mid_x - text.get_width()//2, mid_y - text.get_height()//2))

    def _draw_stats(self, team1, team2):
        """Draw all game statistics under the game frame"""
        # Position under game frame (right side of screen)
        info_x = self.DEBUG_WIDTH + 20
        info_y = self.GAME_HEIGHT + 20

        # Team colors
        color1 = self.COLOR_RED
        color2 = self.COLOR_BLUE

        # Draw game header
        header_text = f"NHL '94 - {'HUMAN' if self.human_control else 'AI'} CONTROL"
        text_surface = self.big_font.render(header_text, True, self.COLOR_WHITE)
        self.screen.blit(text_surface, (info_x, info_y))

        # Draw team stats (full details)
        stats_y = info_y + 30  # Below header

        # Team 1 stats (left column)
        team1_stats = [
            f"TEAM 1:",
            f"Score: {team1.stats.score}",
            f"Shots: {team1.stats.shots}",
            f"Checks: {team1.stats.bodychecks}",
            f"Attack: {team1.stats.attackzone}",
            f"Faceoffs: {team1.stats.faceoffwon}",
            f"Passing: {team1.stats.passing}",
            f"OneTimer: {team1.stats.onetimer}",
            f"Puck: {'Player' if team1.player_haspuck else 'Goalie' if team1.goalie_haspuck else 'None'}"
        ]

        for i, text in enumerate(team1_stats):
            color = color1 if i > 0 else self.COLOR_WHITE  # Header white
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (info_x, stats_y + i * 20))

        # Team 2 stats (right column)
        team2_stats = [
            f"TEAM 2:",
            f"Score: {team2.stats.score}",
            f"Shots: {team2.stats.shots}",
            f"Checks: {team2.stats.bodychecks}",
            f"Attack: {team2.stats.attackzone}",
            f"Faceoffs: {team2.stats.faceoffwon}",
            f"Passing: {team2.stats.passing}",
            f"OneTimer: {team2.stats.onetimer}",
            f"Puck: {'Player' if team2.player_haspuck else 'Goalie' if team2.goalie_haspuck else 'None'}"
        ]

        for i, text in enumerate(team2_stats):
            color = color2 if i > 0 else self.COLOR_WHITE  # Header white
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (info_x + 150, stats_y + i * 20))  # Right column

        # Draw visualization toggle status below stats
        toggle_y = stats_y + len(team1_stats) * 20 + 10

        toggle_text = [
            "VISUALIZATION TOGGLES:",
            f"[1] Passing Lanes: {'ON' if self.show_passing_lanes else 'OFF'}",
            f"[2] Velocities: {'ON' if self.show_velocities else 'OFF'}",
            f"[3] Orientations: {'ON' if self.show_orientations else 'OFF'}",
            f"[4] Distances: {'ON' if self.show_distances else 'OFF'}"
        ]

        for i, text in enumerate(toggle_text):
            text_surface = self.font.render(text, True, self.COLOR_CYAN if i == 0 else self.COLOR_WHITE)
            self.screen.blit(text_surface, (info_x, toggle_y + i * 20))

        # Draw controls help at the very bottom
        controls_y = toggle_y + len(toggle_text) * 20 + 10
        help_text = "CONTROLS: Arrows=Move, X=Slap, C=Wrist, Z=Pass, F1=AI/Human"
        text_surface = self.font.render(help_text, True, self.COLOR_YELLOW)
        self.screen.blit(text_surface, (info_x, controls_y))

    def draw_frame(self, frame_img, info, action=None):
        # Clear surfaces
        self.debug_surf.fill(self.COLOR_BLACK)
        self.game_surf.fill(self.COLOR_BLACK)

        # Draw the rink and game elements
        self._draw_rink()

        # Draw players and passing lanes for both teams
        team1 = self.game_state.team1
        team2 = self.game_state.team2

        # Draw team 1 (red)
        for player in team1.players:
            self._draw_player(player, self.COLOR_RED)
        self._draw_player(team1.goalie, self.COLOR_RED, is_goalie=True)
        if self.show_passing_lanes:
            self._draw_passing_lanes(team1, self.COLOR_RED)
        if self.show_distances:
            self._draw_controlled_distances(team1, self.COLOR_RED)

        # Draw team 2 (blue)
        for player in team2.players:
            self._draw_player(player, self.COLOR_BLUE)
        self._draw_player(team2.goalie, self.COLOR_BLUE, is_goalie=True)
        if self.show_passing_lanes:
            self._draw_passing_lanes(team2, self.COLOR_BLUE)
        if self.show_distances:
            self._draw_controlled_distances(team2, self.COLOR_BLUE)

        # Draw puck
        self._draw_puck(self.game_state.puck)

        # Draw stats and debug info
        self._draw_stats(team1, team2)

        # Draw action info if available
        if action is not None:
            action_names = ["Up", "Down", "Left", "Right", "B", "C", "A", "Start", "Mode", "X", "Y", "Z"]
            active_actions = [name for name, val in zip(action_names, action) if val]
            action_text = "Active: " + ", ".join(active_actions) if active_actions else "No actions"
            text_surface = self.font.render(action_text, True, self.COLOR_WHITE)
            self.debug_surf.blit(text_surface, (self.DEBUG_WIDTH/2 - text_surface.get_width()/2, 120))

        # Draw the actual game frame
        emu_screen = np.transpose(frame_img, (1, 0, 2))
        game_surf = pygame.surfarray.make_surface(emu_screen)
        scaled_game = pygame.transform.scale(game_surf, (self.GAME_WIDTH, self.GAME_HEIGHT))
        self.game_surf.blit(scaled_game, (0, 0))

        # Combine both surfaces on the screen
        self.screen.blit(self.debug_surf, (0, 0))
        self.screen.blit(self.game_surf, (self.DEBUG_WIDTH, 0))

        pygame.display.flip()