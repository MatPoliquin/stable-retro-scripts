"""
Display
"""

from os import environ
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gymnasium as gym
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame # pylint: disable=wrong-import-position
import pygame.freetype # pylint: disable=wrong-import-position

FB_WIDTH = 1920
FB_HEIGHT = 1080

class FullScreenDisplayEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.GAME_WIDTH = int(FB_WIDTH * 0.8)
        self.GAME_HEIGHT = FB_HEIGHT

        # Init Window
        pygame.init()
        self.screen = pygame.display.set_mode((FB_WIDTH, FB_HEIGHT), pygame.FULLSCREEN | pygame.NOFRAME | pygame.SCALED, vsync=1)
        self.main_surf = pygame.Surface((FB_WIDTH, FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, ac):
        ob, rew, done, info = self.env.step(ac)

        framebuffer = self.env.render(mode='rgb_array')

        self.main_surf.fill((0, 0, 0))
        emu_screen = np.transpose(framebuffer, (1,0,2))

        surf = pygame.surfarray.make_surface(emu_screen)

        self.main_surf.set_colorkey(None)
        x_pos = (FB_WIDTH - self.GAME_WIDTH) / 2
        self.main_surf.blit(pygame.transform.scale(surf,(self.GAME_WIDTH, self.GAME_HEIGHT)), (x_pos, 0))

        self.screen.blit(pygame.transform.smoothscale(self.main_surf,(FB_WIDTH, FB_HEIGHT)), (0, 0))

        pygame.display.flip()

        self.get_input()

        keystate = self.get_input()
        if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
            #logger.log('Exiting...')
            exit()

        return ob, rew, done, info

    def seed(self, s):
        self.rng.seed(s)

    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        return keystate