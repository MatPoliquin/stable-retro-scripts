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

# GameDisplay
class GameDisplay:
    def __init__(self, args):
        # Init Window
        pygame.init()
        #screen = pygame.display.set_mode((1920, 1080))
        self.screen = pygame.display.set_mode((960, 540))
        self.main_surf = pygame.Surface((1920, 1080))
        self.main_surf.set_colorkey((0,0,0))
        self.ft_font = pygame.freetype.SysFont('Times New Roman', 80)


    def draw_contact_info(self):
        text_rect = self.ft_font.get_rect('videogames.ai')
        text_rect.center = self.screen.get_rect().center
        self.ft_font.render_to(self.screen, text_rect.topleft, 'videogames.ai', (255, 255, 255))

    def draw_frame(self, frame_img):
        self.main_surf.fill((0, 0, 0))
        emu_screen = np.transpose(frame_img, (1,0,2))
        #print(emu_screen.shape)
        #surf.fill((0,0,0))
        surf = pygame.surfarray.make_surface(emu_screen)

        self.main_surf.blit(pygame.transform.scale(surf,(1920,1080)), (0, 0))
        #print(main_surf.get_colorkey())
        self.main_surf.set_colorkey(None)
        #main_surf.convert()
        self.screen.blit(pygame.transform.smoothscale(self.main_surf,(960,540)), (0, 0))
        #screen.blit(surf, (0, 0))
 
        self.draw_contact_info()

        pygame.display.flip()

    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        return keystate

