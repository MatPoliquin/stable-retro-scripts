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

class PvPGameDisplay:
    def __init__(self, args, total_params, nn_type, button_names):
        self.GAME_WIDTH = 320 * 4
        self.GAME_HEIGHT = 240 * 4
        self.BASIC_INFO_X = (FB_WIDTH >> 1) - 50
        self.BASIC_INFO_Y = self.GAME_HEIGHT + 10
        self.AP_X = self.GAME_WIDTH + 100
        self.AP_Y = 200
        # Init Window
        pygame.init()
        self.screen = pygame.display.set_mode((args.display_width, args.display_height))
        self.main_surf = pygame.Surface((FB_WIDTH, FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))
        self.font = pygame.freetype.SysFont('symbol', 30)
        self.args = args
        self.num_params = total_params
        self.nn_type = nn_type
        self.button_names = button_names

    def draw_string(self, font, str, pos, color):
        text_rect = font.get_rect(str)
        text_rect.topleft = pos
        font.render_to(self.main_surf, text_rect.topleft, str, color)
        return text_rect.bottom

    def draw_contact_info(self):
        text_rect = self.font.get_rect('videogames.ai')
        text_rect.topleft = (FB_WIDTH - text_rect.width, FB_HEIGHT - text_rect.height)
        self.font.render_to(self.main_surf, text_rect.topleft, 'videogames.ai', (255, 255, 255))

    def draw_action_probabilties(self, pos_x, pos_y, action_probabilities):

        y = pos_y + 10
        for button in self.button_names:
            self.draw_string(self.font, button, (pos_x, y), (255, 255, 255))
            y += 30

        y = pos_y + 10
        for prob in action_probabilities:
            self.draw_string(self.font, ('%f' % prob), (pos_x + 150, y), (255, 255, 255))
            y += 30

    def draw_basic_info(self):
        bottom_y = self.draw_string(self.font, ('ENV: %s' % self.args.env), (self.BASIC_INFO_X, self.BASIC_INFO_Y), (255, 255, 255))
        #bottom_y = self.draw_string(self.font, ('MODEL: %s' % self.nn_type), (self.BASIC_INFO_X, bottom_y + 5), (255, 255, 255))
        #bottom_y = self.draw_string(self.font, ('NUM PARAMS:%d' % self.num_params), (self.BASIC_INFO_X, bottom_y + 5), (255, 255, 255))

    def draw_frame(self, frame_img, p1_action_probabilities, p2_action_probabilities):
        self.main_surf.fill((0, 0, 0))
        emu_screen = np.transpose(frame_img, (1,0,2))

        #print(input_state)
        #print(emu_screen.shape)
        #surf.fill((0,0,0))
        surf = pygame.surfarray.make_surface(emu_screen)

        #TODO draw input state
        #input_state_surf = pygame.surfarray.make_surface(input_state)

        game_x = (FB_WIDTH - self.GAME_WIDTH) >> 1
        self.main_surf.blit(pygame.transform.scale(surf,(self.GAME_WIDTH, self.GAME_HEIGHT)), (game_x, 0))

        self.draw_contact_info()
        self.draw_basic_info()
        self.draw_action_probabilties(0, 100, p1_action_probabilities)
        self.draw_action_probabilties(self.GAME_WIDTH + game_x, 100, p2_action_probabilities)
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

class GameDisplay:
    def __init__(self, args, total_params, nn_type, button_names):
        self.GAME_WIDTH = 320 * 4
        self.GAME_HEIGHT = 240 * 4
        self.BASIC_INFO_X = 0
        self.BASIC_INFO_Y = self.GAME_HEIGHT + 10
        self.ENV_X = 100
        self.ENV_Y = FB_HEIGHT - 20
        self.MODELDESC_X = 600
        self.MODELDESC_Y = FB_HEIGHT - 20
        self.NUM_PARAMS_X = 1100
        self.NUM_PARAMS_Y = FB_HEIGHT - 20
        self.AP_X = self.GAME_WIDTH + 100
        self.AP_Y = 200
        # Init Window
        pygame.init()
        self.screen = pygame.display.set_mode((args.display_width, args.display_height))
        self.main_surf = pygame.Surface((FB_WIDTH, FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))
        self.font = pygame.freetype.SysFont('symbol', 30)
        self.info_font = pygame.freetype.SysFont('symbol', 20)
        self.info_font_big = pygame.freetype.SysFont('symbol', 50)
        self.args = args
        self.num_params = total_params
        self.nn_type = nn_type
        self.button_names = button_names

    def draw_string(self, font, str, pos, color):
        text_rect = font.get_rect(str)
        text_rect.topleft = pos
        font.render_to(self.main_surf, text_rect.topleft, str, color)
        return text_rect.bottom

    def draw_contact_info(self):
        text_rect = self.font.get_rect('videogames.ai')
        text_rect.topleft = (FB_WIDTH - text_rect.width, FB_HEIGHT - text_rect.height)
        self.font.render_to(self.main_surf, text_rect.topleft, 'videogames.ai', (255, 255, 255))

    def draw_action_probabilties(self, action_probabilities):

        y = self.AP_Y + 10
        for button in self.button_names:
            self.draw_string(self.font, button, (self.AP_X, y), (255, 255, 255))
            y += 30

        y = self.AP_Y + 10
        for prob in action_probabilities:
            self.draw_string(self.font, ('%f' % prob), (self.AP_X + 150, y), (255, 255, 255))
            y += 30

    def draw_basic_info(self):
        #bottom_y = self.draw_string(self.info_font, ('ENV: %s' % self.args.env), (self.BASIC_INFO_X, self.BASIC_INFO_Y), (255, 255, 255))
        #bottom_y = self.draw_string(self.info_font, ('MODEL: %s' % self.nn_type), (self.BASIC_INFO_X, bottom_y + 5), (255, 255, 255))
        #bottom_y = self.draw_string(self.info_font, ('NUM PARAMS:%d' % self.num_params), (self.BASIC_INFO_X, bottom_y + 5), (255, 255, 255))


        self.draw_string(self.info_font, 'ENV', (self.ENV_X, self.ENV_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'MODEL', (self.MODELDESC_X, self.MODELDESC_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'NUM PARAMETERS', (self.NUM_PARAMS_X, self.NUM_PARAMS_Y), (0, 255, 0))

        self.draw_string(self.font, self.args.env, (self.ENV_X - 100, self.ENV_Y - 70), (255, 255, 255))
        self.draw_string(self.info_font_big, self.nn_type, (self.MODELDESC_X, self.MODELDESC_Y - 70), (255, 255, 255))
        self.draw_string(self.info_font_big, ('%d' % self.num_params), (self.NUM_PARAMS_X, self.NUM_PARAMS_Y - 70), (255, 255, 255))

    def draw_frame(self, frame_img, action_probabilities, input_state):
        self.main_surf.fill((0, 0, 0))
        emu_screen = np.transpose(frame_img, (1,0,2))

        #print(input_state)
        #print(emu_screen.shape)
        #surf.fill((0,0,0))
        surf = pygame.surfarray.make_surface(emu_screen)

        #TODO draw input state
        #input_state_surf = pygame.surfarray.make_surface(input_state)

        self.main_surf.blit(pygame.transform.scale(surf,(self.GAME_WIDTH, self.GAME_HEIGHT)), (0, 0))

        self.draw_contact_info()
        self.draw_basic_info()
        self.draw_action_probabilties(action_probabilities)
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
