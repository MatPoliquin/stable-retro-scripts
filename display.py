"""
Display
"""

import os, datetime
import argparse
import retro
import gym
import numpy as np
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import pygame.freetype

FB_WIDTH = 1920
FB_HEIGHT = 1080
GAME_WIDTH = 1728
GAME_HEIGHT = 864
BASIC_INFO_X = 0
BASIC_INFO_Y = GAME_HEIGHT + 10


# GameDisplay
class GameDisplay:
    def __init__(self, args, total_params=0):
        # Init Window
        pygame.init()
        self.screen = pygame.display.set_mode((args.display_width, args.display_height))
        self.main_surf = pygame.Surface((FB_WIDTH, FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))
        self.font = pygame.freetype.SysFont('symbol', 30)
        self.args = args
        self.num_params = total_params

    def draw_contact_info(self):
        text_rect = self.font.get_rect('videogames.ai')
        text_rect.topleft = (FB_WIDTH - text_rect.width, FB_HEIGHT - text_rect.height)
        self.font.render_to(self.main_surf, text_rect.topleft, 'videogames.ai', (255, 255, 255))

    def draw_string(self, font, str, pos, color):
        text_rect = font.get_rect(str)
        text_rect.topleft = pos
        font.render_to(self.main_surf, text_rect.topleft, str, color)
        return text_rect.bottom

    def draw_basic_info(self):
        bottom_y = self.draw_string(self.font, ('ENV: %s' % self.args.env), (BASIC_INFO_X, BASIC_INFO_Y), (255, 255, 255))
        bottom_y = self.draw_string(self.font, 'MODEL: NATURE CNN', (BASIC_INFO_X, bottom_y + 5), (255, 255, 255))
        bottom_y = self.draw_string(self.font, ('NUM PARAMS:%d' % self.num_params), (BASIC_INFO_X, bottom_y + 5), (255, 255, 255))

    def draw_frame(self, frame_img):
        self.main_surf.fill((0, 0, 0))
        emu_screen = np.transpose(frame_img, (1,0,2))
        #print(emu_screen.shape)
        #surf.fill((0,0,0))
        surf = pygame.surfarray.make_surface(emu_screen)

        self.main_surf.blit(pygame.transform.scale(surf,(GAME_WIDTH, GAME_HEIGHT)), (0, 0))

        self.draw_contact_info()
        self.draw_basic_info()
        #print(main_surf.get_colorkey())
        self.main_surf.set_colorkey(None)
        #main_surf.convert()
        self.screen.blit(pygame.transform.smoothscale(self.main_surf,(self.args.display_width,self.args.display_height)), (0, 0))
        #screen.blit(surf, (0, 0))
 
        pygame.display.flip()

    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        return keystate

