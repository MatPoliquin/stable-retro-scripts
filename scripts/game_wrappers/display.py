"""
Display
Adapted from NHL94 display. TODO clean-up
"""

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # must be set before pygame import
import gymnasium as gym
import numpy as np
from collections import deque
import pygame
import pygame.freetype

class GameDisplayEnv(gym.Wrapper):
    FB_WIDTH = 1920
    FB_HEIGHT = 1080
    GAME_WIDTH = 320 * 4
    GAME_HEIGHT = 240 * 4

    FONT_NAME = 'symbol'
    FONT_SIZE = 30
    INFO_FONT_SIZE = 20
    INFO_FONT_BIG_SIZE = 50

    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_CYAN = (0, 255, 255)
    COLOR_BLUE = (0, 0, 255)
    COLOR_LIGHT_GRAY = (50, 50, 50)
    COLOR_BACKGROUND = (30, 30, 30)

    MAX_REWARDS = 200

    def __init__(self, env, args, total_params, nn_type, button_names):
        self.env = env

        # Layout positions as a dict for readability
        self.positions = {
            "basic_info": (0, self.GAME_HEIGHT + 10),
            "env": (100, self.FB_HEIGHT - 20),
            "model_desc": (600, self.FB_HEIGHT - 20),
            "num_params": (1100, self.FB_HEIGHT - 20),
            "ap_title": (self.GAME_WIDTH + 300, 200),
            "input_title": (self.GAME_WIDTH + 10, 200),
            "ai": (self.GAME_WIDTH + 10, 20),
            "stats": (600, self.GAME_HEIGHT + 10),
            "rf": (self.GAME_WIDTH + 10, 700),
            "ap": (self.GAME_WIDTH + 300, 200),
            "input": (self.GAME_WIDTH + 10, 200),
        }

        # Now that self.positions exists, define env_label based on existing entries
        self.positions["env_label"] = (self.positions["env"][0] - 100, self.positions["env"][1] - 70)

        pygame.init()
        flags = pygame.RESIZABLE | (pygame.FULLSCREEN if args.fullscreen else 0)
        self.screen = pygame.display.set_mode((self.FB_WIDTH, self.FB_HEIGHT), flags)
        self.main_surf = pygame.Surface((self.FB_WIDTH, self.FB_HEIGHT))
        self.main_surf.set_colorkey((0, 0, 0))

        self.font = pygame.freetype.SysFont(self.FONT_NAME, self.FONT_SIZE)
        self.font.antialiased = True
        self.info_font = pygame.freetype.SysFont(self.FONT_NAME, self.INFO_FONT_SIZE)
        self.info_font_big = pygame.freetype.SysFont(self.FONT_NAME, self.INFO_FONT_BIG_SIZE)

        self.args = args
        self.num_params = total_params
        self.nn_type = nn_type
        self.button_names = button_names

        self.scr_count = 0
        self.action_probabilities = None
        self.player_actions = [0] * 12
        self.frameRewardList = deque([0.0] * self.MAX_REWARDS, maxlen=self.MAX_REWARDS)

        #self.game_state = NHL94GameState(1 if args.env == 'NHL941on1-Genesis' else 2)

        self.model_in_use = 0
        self.model_params = [None, None, None]
        self.model_names = ["CODE", "DEFENSE ZONE", "SCORING OPP"]

        # Key to player action index mapping
        self.key_action_map = {
            pygame.K_x: 0,
            pygame.K_z: 1,
            pygame.K_TAB: 2,
            pygame.K_RETURN: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_c: 8,
            pygame.K_a: 9,
            pygame.K_s: 10,
            pygame.K_d: 11,
        }

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def set_reward(self, reward: float) -> None:
        """Append reward to frame reward list."""
        self.reward = reward
        self.frameRewardList.append(reward)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.set_reward(rew[0])

        framebuffer = self.render()

        self.draw_frame(framebuffer, None, obs, info)

        return obs, rew, done, info

    def seed(self, seed_val):
        # Presuming seedable RNG exists (not shown in code)
        if hasattr(self, 'rng'):
            self.rng.seed(seed_val)

    def draw_string(self, font, text: str, pos, color):
        rect = font.get_rect(text)
        rect.topleft = pos
        font.render_to(self.screen, rect.topleft, text, color)
        return rect.bottom

    def draw_contact_info(self):
        text = 'stable-retro'
        rect = self.font.get_rect(text)
        rect.topleft = (self.FB_WIDTH - rect.width, self.FB_HEIGHT - rect.height)
        self.font.render_to(self.screen, rect.topleft, text, self.COLOR_WHITE)

    def draw_basic_info(self):
        env_x, env_y = self.positions['env']
        model_x, model_y = self.positions['model_desc']
        # num_params = self.positions['num_params'] # commented out in original code

        self.draw_string(self.info_font, 'ENV', (env_x, env_y), self.COLOR_GREEN)
        self.draw_string(self.info_font, 'GAME STATS', (model_x, model_y), self.COLOR_GREEN)

        # Display env name
        env_label_x, env_label_y = env_x - 100, env_y - 70
        self.draw_string(self.font, self.args.env, (env_label_x, env_label_y), self.COLOR_WHITE)

    def draw_ai_overview(self):
        return
        ai_x, ai_y = self.positions['ai']
        pygame.draw.rect(self.screen, self.COLOR_WHITE, pygame.Rect(ai_x - 5, ai_y - 5, 580, 100), width=3)

        self.draw_string(self.info_font, 'MODEL(TYPE): NUM PARAMS', (ai_x, ai_y), self.COLOR_GREEN)

        for i, (name, params) in enumerate(zip(self.model_names, self.model_params)):
            num_params = 'N/A' if params is None else params
            color = self.COLOR_BLUE if self.model_in_use == i else self.COLOR_WHITE
            self.draw_string(self.info_font, f'{name}: {num_params}', (ai_x, ai_y + 20 + i * 20), color)

    def draw_model(self, input_state):
        return

        input_title_x, input_title_y = self.positions['input_title']

        self.draw_string(self.info_font, f'CURRENT MODEL: {self.model_names[self.model_in_use]}',
                         (input_title_x, input_title_y - 25), self.COLOR_GREEN)

        pygame.draw.rect(self.screen, self.COLOR_WHITE,
                         pygame.Rect(input_title_x - 5, input_title_y - 5, 580, 470), width=3)

        self.draw_string(self.info_font, 'INPUT', (input_title_x, input_title_y + 20), self.COLOR_GREEN)

        color = self.COLOR_LIGHT_GRAY if self.model_in_use == 0 else self.COLOR_WHITE

        # Looping over input_state elements to avoid repetition
        labels = [
            ('P1 POS', [0, 1]),
            ('P1 VEL', [2, 3]),
            ('P2 POS', [4, 5]),
            ('P2 VEL', [6, 7]),
            ('PUCK POS', [8, 9]),
            ('PUCK VEL', [10, 11]),
            ('G2 POS', [12, 13]),
            ('P1/G1 HASPUCK', [14, 15]),
        ]
        if input_state is not None:
            for i, (label, idxs) in enumerate(labels):
                vals = (input_state[0][idxs[0]], input_state[0][idxs[1]])
                self.draw_string(self.info_font, f'{label}: ({vals[0]:.3f}, {vals[1]:.3f})',
                                 (input_title_x, input_title_y + 40 + i * 20), color)

        ap_title_x, ap_title_y = self.positions['ap_title']
        self.draw_string(self.info_font, 'OUTPUT', (ap_title_x, ap_title_y + 20), self.COLOR_GREEN)
        self.draw_string(self.info_font, 'Action            Confidence', (ap_title_x, ap_title_y + 40), self.COLOR_CYAN)

        if self.action_probabilities is None:
            return

        y = ap_title_y + 60
        for button in self.button_names:
            self.draw_string(self.font, button, (ap_title_x + 10, y), color)
            y += 30

        y = ap_title_y + 60
        for prob in self.action_probabilities:
            self.draw_string(self.font, f'{prob:.6f}', (ap_title_x + 150, y), color)
            y += 30

    def draw_game_stats(self, info):
        return
        if not info:
            return
        stats_x, stats_y = self.positions['stats']

        stats = [
            ('P1 SHOTS', 'p1_shots', 0, 0),
            ('P2 SHOTS', 'p2_shots', 300, 0),
            ('P1 ATKZONE', 'p1_attackzone', 0, 20),
            ('P2 ATKZONE', 'p2_attackzone', 300, 20),
            ('P1 BODYCHECKS', 'p1_passing', 0, 40),
            ('P2 BODYCHECKS', 'p2_passing', 300, 40),
            ('P1 FACEOFFWON', 'p1_faceoffwon', 0, 60),
            ('P2 FACEOFFWON', 'p2_faceoffwon', 300, 60),
        ]
        # info is a list, original code accesses info[0]
        info_dict = info[0]

        for label, key, x_off, y_off in stats:
            value = info_dict.get(key, 0)
            self.draw_string(self.info_font, f'{label}: {value}', (stats_x + x_off, stats_y + y_off), self.COLOR_CYAN)

    def set_ai_sys_info(self, ai_sys):
        if ai_sys is None:
            return
        self.action_probabilities = ai_sys.display_probs
        self.model_params = ai_sys.model_num_params
        #self.model_in_use = ai_sys.model_in_use

    def draw_frame_reward_histogram(self, pos_x, pos_y, width, height):
        if not self.frameRewardList:
            return

        last_reward = self.frameRewardList[-1]
        #print(self.frameRewardList)
        #print(last_reward)
        self.draw_string(self.info_font, f'REWARD FUNCTION: {last_reward:.6f}', (pos_x, pos_y - 5), self.COLOR_BLUE)

        bar_height = 50
        base_line_y = pos_y + 100
        pygame.draw.line(self.screen, self.COLOR_WHITE, (pos_x, base_line_y), (pos_x + 600, base_line_y))
        pygame.draw.line(self.screen, self.COLOR_WHITE,
                         (pos_x, base_line_y - bar_height), (pos_x, base_line_y + bar_height))

        for i, r in enumerate(self.frameRewardList):
            x = pos_x + i * 3
            height = abs(r) * bar_height
            y = base_line_y if r < 0 else base_line_y - height
            if r != 0:
                pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(x, y, 2, height))

    def draw_frame(self, frame_img, action_probabilities, input_state, info):
        self.screen.fill(self.COLOR_BACKGROUND)

        # Draw game frame: note frame_img axes order
        emu_screen = np.transpose(frame_img, (1, 0, 2))
        surf = pygame.surfarray.make_surface(emu_screen)
        self.screen.blit(pygame.transform.scale(surf, (self.GAME_WIDTH, self.GAME_HEIGHT)), (0, 0))

        self.draw_contact_info()
        self.draw_basic_info()
        self.draw_model(input_state)
        self.draw_frame_reward_histogram(self.positions['rf'][0], self.positions['rf'][1], 100, 20)
        self.draw_game_stats(info)
        self.draw_ai_overview()

        pygame.display.flip()

        keystate = self.get_input()
        self.process_key_state(keystate)

    def process_key_state(self, keystate):
        if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
            raise SystemExit("User requested exit")

        for key, idx in self.key_action_map.items():
            self.player_actions[idx] = 1 if keystate[key] else 0

        if keystate[pygame.K_i]:
            self.scr_count += 1
            pygame.image.save(self.screen, f"screenshot{self.scr_count:02d}.png")

    def get_input(self):
        pygame.event.pump()
        return pygame.key.get_pressed()
