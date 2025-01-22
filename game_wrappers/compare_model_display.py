"""
Display
"""

import os
import datetime
import math
import argparse
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import pygame.freetype
import cv2
import gymnasium as gym
import numpy as np
import retro

FB_WIDTH = 1920
FB_HEIGHT = 1080

class CompareModelDisplay():
    def __init__(self, args, model1_desc, model2_desc, model1_params, model2_params, button_names):

        self.GAME_WIDTH = int(320 * 2.8)
        self.GAME_HEIGHT = int(240 * 2.8)
        self.BASIC_INFO_X = (FB_WIDTH >> 1) - 50
        self.BASIC_INFO_Y = self.GAME_HEIGHT + 10
        self.AP_X = self.GAME_WIDTH + 100
        self.AP_Y = 200
        self.SCREEN_OFFSET = 50
        self.MODELDESC1_X = 100
        self.MODELDESC1_Y = FB_HEIGHT - 100
        self.NUM_PARAMS1_X = self.MODELDESC1_X + 200
        self.NUM_PARAMS1_Y = FB_HEIGHT - 100
        self.MODELDESC2_X = FB_WIDTH - 200
        self.MODELDESC2_Y = FB_HEIGHT - 100
        self.NUM_PARAMS2_X = self.MODELDESC2_X - 350
        self.NUM_PARAMS2_Y = FB_HEIGHT - 100
        self.VS_X = (FB_WIDTH >> 1) - 50
        self.VS_Y = FB_HEIGHT - 150

        # Init Window
        pygame.init()
        self.screen = pygame.display.set_mode((args.display_width, args.display_height))
        self.main_surf = pygame.Surface((FB_WIDTH, FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))
        self.font = pygame.freetype.SysFont('symbol', 30)
        self.info_font = pygame.freetype.SysFont('symbol', 20)
        self.info_font_big = pygame.freetype.SysFont('symbol', 50)
        self.vs_font = pygame.freetype.SysFont('symbol', 80)
        self.args = args
        self.button_names = button_names
        self.model1_desc = model1_desc
        self.model2_desc = model2_desc
        self.model1_params = model1_params
        self.model2_params = model2_params
        self.p1_action_probabilities = [0] * 12
        self.p2_action_probabilities = [0] * 12

    def draw_string(self, font, str, pos, color):
        text_rect = font.get_rect(str)
        text_rect.topleft = pos
        font.render_to(self.main_surf, text_rect.topleft, str, color)
        return text_rect.bottom

    def draw_contact_info(self):
        text_rect = self.font.get_rect('stable-retro')
        text_rect.topleft = (FB_WIDTH - text_rect.width, FB_HEIGHT - text_rect.height)
        self.font.render_to(self.main_surf, text_rect.topleft, 'stable-retro', (255, 255, 255))

    def draw_action_probabilties(self, pos_x, pos_y, action_probabilities):

        #print(self.button_names)
        y = pos_y + 10
        for button in self.button_names:
            self.draw_string(self.font, button, (pos_x, y), (255, 255, 255))
            y += 30

        y = pos_y + 10
        for prob in action_probabilities:
            self.draw_string(self.font, ('%f' % prob), (pos_x + 150, y), (255, 255, 255))
            y += 30

    def draw_basic_info(self):
        bottom_y = self.draw_string(self.vs_font, 'VS', (self.VS_X, self.VS_Y), (0, 255, 0))
        bottom_y = self.draw_string(self.font, ('ENV:%s' % self.args.env), (self.VS_X - 100, 10), (255, 255, 255))

        # Model 1
        self.draw_string(self.info_font, 'MODEL', (self.MODELDESC1_X, self.MODELDESC1_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'NUM PARAMETERS', (self.NUM_PARAMS1_X, self.NUM_PARAMS1_Y), (0, 255, 0))

        self.draw_string(self.info_font_big, self.model1_desc, (self.MODELDESC1_X, self.MODELDESC1_Y - 60), (255, 255, 255))
        self.draw_string(self.info_font_big, ('%d' % self.model1_params), (self.NUM_PARAMS1_X, self.NUM_PARAMS1_Y - 60), (255, 255, 255))

        # Model 2
        self.draw_string(self.info_font, 'MODEL', (self.MODELDESC2_X, self.MODELDESC2_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'NUM PARAMETERS', (self.NUM_PARAMS2_X, self.NUM_PARAMS2_Y), (0, 255, 0))

        self.draw_string(self.info_font_big, self.model2_desc, (self.MODELDESC2_X, self.MODELDESC2_Y - 60), (255, 255, 255))
        self.draw_string(self.info_font_big, ('%d' % self.model2_params), (self.NUM_PARAMS2_X, self.NUM_PARAMS2_Y - 60), (255, 255, 255))

    def ProcessKeyState(self, keystate):

        if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
            #logger.log('Exiting...')
            exit()


    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        return keystate


    def run_frame(self, left_frame, right_frame):
        keystate = self.get_input()
        self.ProcessKeyState(keystate)

        self.main_surf.fill((0, 0, 0))


        #draw left screen
        left_screen = np.transpose(left_frame, (1,0,2))

        surf = pygame.surfarray.make_surface(left_screen)

        game_x = 50
        self.main_surf.blit(pygame.transform.scale(surf,(self.GAME_WIDTH, self.GAME_HEIGHT)), (game_x, 50))

        #draw right screen
        right_screen = np.transpose(right_frame, (1,0,2))

        surf = pygame.surfarray.make_surface(right_screen)

        game_x = FB_WIDTH - 50 - (self.GAME_WIDTH)
        self.main_surf.blit(pygame.transform.scale(surf,(self.GAME_WIDTH, self.GAME_HEIGHT)), (game_x, 50))

        self.draw_contact_info()
        self.draw_basic_info()
        #self.draw_action_probabilties(0, 100, self.p1_action_probabilities)
        #self.draw_action_probabilties(self.GAME_WIDTH + game_x, 100, self.p2_action_probabilities)
        self.main_surf.set_colorkey(None)
        self.screen.blit(pygame.transform.smoothscale(self.main_surf,(self.args.display_width,self.args.display_height)), (0, 0))

        pygame.display.flip()
