"""
Display
"""

import os, datetime
import argparse
import retro
import gymnasium as gym
import numpy as np
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import pygame.freetype
import cv2
import math

FB_WIDTH = 1920
FB_HEIGHT = 1080

class NHL94PvPGameDisplayEnv(gym.Wrapper):
    def __init__(self, env, args, model1_desc, model2_desc, model1_params, model2_params, button_names):
        gym.Wrapper.__init__(self, env)

        self.GAME_WIDTH = 320 * 4
        self.GAME_HEIGHT = 240 * 4
        self.BASIC_INFO_X = (FB_WIDTH >> 1) - 50
        self.BASIC_INFO_Y = self.GAME_HEIGHT + 10
        self.AP_X = self.GAME_WIDTH + 100
        self.AP_Y = 200
        self.MODELDESC1_X = (FB_WIDTH - self.GAME_WIDTH) >> 1
        self.MODELDESC1_Y = FB_HEIGHT - 20
        self.NUM_PARAMS1_X = self.MODELDESC1_X + 200
        self.NUM_PARAMS1_Y = FB_HEIGHT - 20
        self.MODELDESC2_X = FB_WIDTH - ((FB_WIDTH - self.GAME_WIDTH) >> 1) - 50
        self.MODELDESC2_Y = FB_HEIGHT - 20
        self.NUM_PARAMS2_X = self.MODELDESC2_X - 350
        self.NUM_PARAMS2_Y = FB_HEIGHT - 20
        self.VS_X = (FB_WIDTH >> 1) - 50
        self.VS_Y = FB_HEIGHT - 100

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
        text_rect = self.font.get_rect('videogames.ai')
        text_rect.topleft = (FB_WIDTH - text_rect.width, FB_HEIGHT - text_rect.height)
        self.font.render_to(self.main_surf, text_rect.topleft, 'videogames.ai', (255, 255, 255))

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
        bottom_y = self.draw_string(self.font, self.args.env, (self.VS_X - 100, FB_HEIGHT - 30), (255, 255, 255))

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

       

    def draw_frame(self, frame_img):
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
        self.draw_action_probabilties(0, 100, self.p1_action_probabilities)
        self.draw_action_probabilties(self.GAME_WIDTH + game_x, 100, self.p2_action_probabilities)
        #print(main_surf.get_colorkey())
        self.main_surf.set_colorkey(None)
        #main_surf.convert()
        self.screen.blit(pygame.transform.smoothscale(self.main_surf,(self.args.display_width,self.args.display_height)), (0, 0))
        #screen.blit(surf, (0, 0))
 
        pygame.display.flip()
    
    def ProcessKeyState(self, keystate):

        if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
            #logger.log('Exiting...')
            exit()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, ac):
        ob, rew, done, info = self.env.step(ac)

        framebuffer = self.render()

        #print(framebuffer)

        #print('TEST')
        #self.action_probabilities = ac
        self.draw_frame(framebuffer)


        self.get_input()

        keystate = self.get_input()
        self.ProcessKeyState(keystate)
       
        return ob, rew, done, info

    def seed(self, s):
        self.rng.seed(s)

    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        return keystate

class NHL94GameDisplayEnv(gym.Wrapper):
    def __init__(self, env, args, total_params, nn_type, button_names):
        gym.Wrapper.__init__(self, env)

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
        self.AP_X = self.GAME_WIDTH + 300
        self.AP_Y = 60
        self.INPUT_X = self.GAME_WIDTH + 10
        self.INPUT_Y = 80
        self.AP_TITLE_X = self.GAME_WIDTH + 300
        self.AP_TITLE_Y = 10
        self.INPUT_TITLE_X = self.GAME_WIDTH + 10
        self.INPUT_TITLE_Y = 10
        self.STATS_X = self.GAME_WIDTH + 10
        self.STATS_Y = 600


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

        self.action_probabilities = None
        self.player_actions = [0] * 12


        self.best_dist = 0


    def reset(self, **kwargs):
        #print(**kwargs)
        return self.env.reset(**kwargs)
        #return self.env.reset()

    def step(self, ac):
        ob, rew, done, info = self.env.step(ac)

        framebuffer = self.render()

        #print(framebuffer)

        #print('TEST')
        #self.action_probabilities = ac
        self.draw_frame(framebuffer, None, ob, info)
       
        return ob, rew, done, info

    def seed(self, s):
        self.rng.seed(s)

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
        self.draw_string(self.info_font, 'OUTPUT', (self.AP_TITLE_X, self.AP_TITLE_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'Action          Confidence', (self.AP_TITLE_X, self.AP_TITLE_Y + 20), (0, 255, 255))


        if action_probabilities is None:
            return

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

    def draw_input(self, input_state):
        self.draw_string(self.info_font, 'INPUT', (self.INPUT_TITLE_X, self.INPUT_TITLE_Y), (0, 255, 0))
        #self.draw_string(self.info_font, '84x84 pixels', (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 20), (0, 255, 255))
        #self.draw_string(self.info_font, 'last 4 frames', (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 40), (0, 255, 255))


        #print(input_state)

        #img = np.array(input_state)

        #frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        #surf = pygame.surfarray.make_surface(frame)
        #self.main_surf.blit(pygame.transform.rotozoom(surf, -90, 3), (self.INPUT_X, self.INPUT_Y))

    def draw_game_stats(self, info):
        print(info)
        p1_x = info.get('p1_x')
        p1_y = info.get('p1_y')
        puck_x = info.get('puck_x')
        puck_y = info.get('puck_y')

        tmp = (p1_x - puck_x)**2 + (p1_y - puck_y)**2
        distance = math.sqrt(tmp)


        if distance > self.best_dist:
            self.best_dist = distance
            #print(self.best_dist)

        self.draw_string(self.info_font, 'GAME STATS', (self.STATS_X, self.STATS_Y), (0, 255, 0))

        self.draw_string(self.info_font, ('P1 SHOTS: %d' %  info.get('p1_shots')), (self.STATS_X, self.STATS_Y + 40), (0, 255, 255))
        self.draw_string(self.info_font, ('P2 SHOTS: %d' %  info.get('p2_shots')), (self.STATS_X + 300, self.STATS_Y + 40), (0, 255, 255))
        self.draw_string(self.info_font, ('P1 PASSES: %d' %  info.get('p1_passing')), (self.STATS_X, self.STATS_Y + 60), (0, 255, 255))
        self.draw_string(self.info_font, ('P2 PASSES: %d' %  info.get('p2_passing')), (self.STATS_X + 300, self.STATS_Y + 60), (0, 255, 255))
        self.draw_string(self.info_font, ('P1 BODYCHECKS: %d' %  info.get('p1_passing')), (self.STATS_X, self.STATS_Y + 80), (0, 255, 255))
        self.draw_string(self.info_font, ('P2 BODYCHECKS: %d' %  info.get('p2_passing')), (self.STATS_X + 300, self.STATS_Y + 80), (0, 255, 255))
        self.draw_string(self.info_font, ('P1 FACEOFFWON: %d' %  info.get('p1_faceoffwon')), (self.STATS_X, self.STATS_Y + 100), (0, 255, 255))
        self.draw_string(self.info_font, ('P2 FACEOFFWON: %d' %  info.get('p2_faceoffwon')), (self.STATS_X + 300, self.STATS_Y + 100), (0, 255, 255))
        self.draw_string(self.info_font, ('PUCK DIST: %f' %  distance), (self.STATS_X + 300, self.STATS_Y + 120), (0, 255, 255))


    def draw_frame(self, frame_img, action_probabilities, input_state, info):
        self.main_surf.fill((30, 30, 30))
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
        self.draw_input(input_state)
        self.draw_action_probabilties(self.action_probabilities)
        #self.draw_game_stats(info)
        #print(main_surf.get_colorkey())
        self.main_surf.set_colorkey(None)
        #main_surf.convert()
        self.screen.blit(pygame.transform.smoothscale(self.main_surf,(self.args.display_width,self.args.display_height)), (0, 0))
        #screen.blit(surf, (0, 0))
 
        pygame.display.flip()

        self.get_input()

        keystate = self.get_input()
        self.ProcessKeyState(keystate)
        

    def ProcessKeyState(self, keystate):

        if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
            #logger.log('Exiting...')
            exit()

        #if keystate[pygame.K_UP]:

        self.player_actions[0] = 1 if keystate[pygame.K_x] else 0
        self.player_actions[1] = 1 if keystate[pygame.K_z] else 0
        self.player_actions[2] = 1 if keystate[pygame.K_TAB] else 0
        self.player_actions[3] = 1 if keystate[pygame.K_RETURN] else 0
        self.player_actions[4] = 1 if keystate[pygame.K_UP] else 0
        self.player_actions[5] = 1 if keystate[pygame.K_DOWN] else 0
        self.player_actions[6] = 1 if keystate[pygame.K_LEFT] else 0
        self.player_actions[7] = 1 if keystate[pygame.K_RIGHT] else 0
        self.player_actions[8] = 1 if keystate[pygame.K_c] else 0
        self.player_actions[9] = 1 if keystate[pygame.K_a] else 0
        self.player_actions[10] = 1 if keystate[pygame.K_s] else 0
        self.player_actions[11] = 1 if keystate[pygame.K_d] else 0

        #if keystate[pygame.K_DOWN]:
        #    print("HELLOOOOOOOO")


    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        return keystate

