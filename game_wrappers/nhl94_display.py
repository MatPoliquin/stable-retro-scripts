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
import sys
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use('Agg')



class NHL94PvPGameDisplayEnv(gym.Wrapper):
    def __init__(self, env, args, model1_desc, model2_desc, model1_params, model2_params, button_names):
        gym.Wrapper.__init__(self, env)


        self.FB_WIDTH = args.display_width
        self.FB_HEIGHT = args.display_height

        self.GAME_WIDTH = 320 * 4
        self.GAME_HEIGHT = 240 * 4
        self.BASIC_INFO_X = (self.FB_WIDTH >> 1) - 50
        self.BASIC_INFO_Y = self.GAME_HEIGHT + 10
        self.AP_X = self.GAME_WIDTH + 100
        self.AP_Y = 200
        self.MODELDESC1_X = (self.FB_WIDTH - self.GAME_WIDTH) >> 1
        self.MODELDESC1_Y = self.FB_HEIGHT - 20
        self.NUM_PARAMS1_X = self.MODELDESC1_X + 200
        self.NUM_PARAMS1_Y = self.FB_HEIGHT - 20
        self.MODELDESC2_X = self.FB_WIDTH - ((self.FB_WIDTH - self.GAME_WIDTH) >> 1) - 50
        self.MODELDESC2_Y = self.FB_HEIGHT - 20
        self.NUM_PARAMS2_X = self.MODELDESC2_X - 350
        self.NUM_PARAMS2_Y = self.FB_HEIGHT - 20
        self.VS_X = (self.FB_WIDTH >> 1) - 50
        self.VS_Y = self.FB_HEIGHT - 100

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
        text_rect.topleft = (self.FB_WIDTH - text_rect.width, self.FB_HEIGHT - text_rect.height)
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
        bottom_y = self.draw_string(self.font, self.args.env, (self.VS_X - 100, self.FB_HEIGHT - 30), (255, 255, 255))

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

        surf = pygame.surfarray.make_surface(emu_screen)

        #TODO draw input state
        #input_state_surf = pygame.surfarray.make_surface(input_state)

        game_x = (self.FB_WIDTH - self.GAME_WIDTH) >> 1
        #self.main_surf.blit(pygame.transform.scale(surf,(self.GAME_WIDTH, self.GAME_HEIGHT)), (game_x, 0))
        self.screen.blit(pygame.transform.smoothscale(surf,(self.args.display_width,self.args.display_height)), (0, 0))

        #self.draw_contact_info()
        #self.draw_basic_info()
        #self.draw_action_probabilties(0, 100, self.p1_action_probabilities)
        #self.draw_action_probabilties(self.GAME_WIDTH + game_x, 100, self.p2_action_probabilities)
        #self.main_surf.set_colorkey(None)
        
 
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

        self.FB_WIDTH = 1920
        self.FB_HEIGHT = 1080

        self.GAME_WIDTH = 320 * 4
        self.GAME_HEIGHT = 240 * 4
        self.BASIC_INFO_X = 0
        self.BASIC_INFO_Y = self.GAME_HEIGHT + 10
        self.ENV_X = 100
        self.ENV_Y = self.FB_HEIGHT - 20
        self.MODELDESC_X = 600
        self.MODELDESC_Y = self.FB_HEIGHT - 20
        self.NUM_PARAMS_X = 1100
        self.NUM_PARAMS_Y = self.FB_HEIGHT - 20
        self.AP_X = self.GAME_WIDTH + 300
        self.AP_Y = 200
        self.INPUT_X = self.GAME_WIDTH + 10
        self.INPUT_Y = 200
        self.AI_X = self.GAME_WIDTH + 10
        self.AI_Y = 20
        self.AP_TITLE_X = self.GAME_WIDTH + 300
        self.AP_TITLE_Y = 200
        self.INPUT_TITLE_X = self.GAME_WIDTH + 10
        self.INPUT_TITLE_Y = 200
        self.STATS_X = self.GAME_WIDTH + 10
        self.STATS_Y = 600

        

        # Init Window
        pygame.init()
        #pygame.joystick.init()
        #joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        #print(joysticks)
        flags = pygame.RESIZABLE
        if args.fullscreen:
            flags |= pygame.FULLSCREEN
        
        self.screen = pygame.display.set_mode((self.FB_WIDTH, self.FB_HEIGHT), flags)
        self.main_surf = pygame.Surface((self.FB_WIDTH, self.FB_HEIGHT))
        self.main_surf.set_colorkey((0,0,0))
        self.font = pygame.freetype.SysFont('symbol', 30)
        self.font.antialiased = True
        self.info_font = pygame.freetype.SysFont('symbol', 20)
        self.info_font_big = pygame.freetype.SysFont('symbol', 50)
        self.args = args
        self.num_params = total_params
        self.nn_type = nn_type
        self.button_names = button_names

        self.action_probabilities = None
        self.player_actions = [0] * 12

        self.best_dist = 0

        #self.ai_sys = None
        self.model_1_num_params = 0
        self.model_2_num_params = 0
        self.model_in_use = 0
        self.model_name = "N/A"

        self.frameRewardList = [10] * 200
        # self.fig = plt.figure(0)
        # self.fig.set_facecolor('black')

        # numYData = len(self.frameRewardList)
        # plt.xlim([0,numYData])
        # plt.ylim([-1,1])
        # plt.tight_layout()
  
        # plt.grid(True)
        # plt.rc('grid', color='w', linestyle='solid')


        # self.fig.set_size_inches(200/80, 20/60, forward=True)

        # ax = plt.gca()
        # ax.set_facecolor("black")

        # ax.tick_params(axis='x', colors='green')
        # ax.tick_params(axis='y', colors='green')

        # ax.get_xaxis().set_ticks([])

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, ac):
        ob, rew, done, info = self.env.step(ac)

        framebuffer = self.render()

        self.draw_frame(framebuffer, None, ob, info)
       
        return ob, rew, done, info

    def seed(self, s):
        self.rng.seed(s)

    def draw_string(self, font, str, pos, color):
        text_rect = font.get_rect(str)
        text_rect.topleft = pos
        font.render_to(self.screen, text_rect.topleft, str, color)
        return text_rect.bottom

    def draw_contact_info(self):
        text_rect = self.font.get_rect('stable-retro')
        text_rect.topleft = (self.FB_WIDTH - text_rect.width, self.FB_HEIGHT - text_rect.height)
        self.font.render_to(self.screen, text_rect.topleft, 'stable-retro', (255, 255, 255))

    def draw_basic_info(self):
        self.draw_string(self.info_font, 'ENV', (self.ENV_X, self.ENV_Y), (0, 255, 0))
        self.draw_string(self.info_font, 'AI', (self.MODELDESC_X, self.MODELDESC_Y), (0, 255, 0))
        #self.draw_string(self.info_font, 'NUM PARAMETERS', (self.NUM_PARAMS_X, self.NUM_PARAMS_Y), (0, 255, 0))

        self.draw_string(self.font, self.args.env, (self.ENV_X - 100, self.ENV_Y - 70), (255, 255, 255))
        self.draw_string(self.info_font_big, '2xMLP', (self.MODELDESC_X, self.MODELDESC_Y - 70), (255, 255, 255))
        #self.draw_string(self.info_font_big, ('%d' % self.num_params), (self.NUM_PARAMS_X, self.NUM_PARAMS_Y - 70), (255, 255, 255))

    def draw_ai_overview(self):

        pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(self.AI_X - 5, self.AI_Y - 5, self.AI_X + 200, self.AI_Y + 70), width=3)

        self.draw_string(self.info_font, 'MODEL(TYPE): NUM PARAMS', (self.AI_X, self.AI_Y), (0, 255, 0))

        color = [(255, 255, 255), (255, 255, 255), (255, 255, 255)]
        
        color[self.model_in_use] = (0, 0, 255)


        self.draw_string(self.info_font, ('CODE: N/A'), (self.AI_X, self.AI_Y + 20), color[0])
        self.draw_string(self.info_font, (('SCORING OPP (MLP): %d') % self.model_2_num_params), (self.AI_X, self.AI_Y + 40), color[2])
        self.draw_string(self.info_font, (('DEFENSE ZONE (MLP): %d') % self.model_1_num_params), (self.AI_X, self.AI_Y + 60), color[1])




    def draw_model(self, input_state):
        self.draw_string(self.info_font, ('CURRENT MODEL:%s' % self.model_name), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y - 25), (0, 255, 0))
        pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(self.INPUT_TITLE_X - 5, self.INPUT_TITLE_Y - 5, self.INPUT_TITLE_X + 200, self.INPUT_TITLE_Y + 260), width=3)

        
        self.draw_string(self.info_font, 'INPUT', (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 20), (0, 255, 0))

        self.draw_string(self.info_font, ('P1 POS: (%.3f, %.3f)' % (input_state[0][0], input_state[0][1])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 40), (255, 255, 255))
        self.draw_string(self.info_font, ('P1 VEL: (%.3f, %.3f)' % (input_state[0][2], input_state[0][3])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 60), (255, 255, 255))
        self.draw_string(self.info_font, ('P2 POS: (%.3f, %.3f)' % (input_state[0][4], input_state[0][5])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 80), (255, 255, 255))
        self.draw_string(self.info_font, ('P2 VEL: (%.3f, %.3f)' % (input_state[0][6], input_state[0][7])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 100), (255, 255, 255))
        self.draw_string(self.info_font, ('PUCK POS: (%.3f, %.3f)' % (input_state[0][8], input_state[0][9])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 120), (255, 255, 255))
        self.draw_string(self.info_font, ('PUCK VEL: (%.3f, %.3f)' % (input_state[0][10], input_state[0][11])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 140), (255, 255, 255))
        self.draw_string(self.info_font, ('G2 POS: (%.3f, %.3f)' % (input_state[0][12], input_state[0][13])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 160), (255, 255, 255))
        self.draw_string(self.info_font, ('P1/G1 HASPUCK: (%.1f, %.1f)' % (input_state[0][14], input_state[0][15])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 180), (255, 255, 255))

        self.draw_string(self.info_font, 'OUTPUT', (self.AP_TITLE_X, self.AP_TITLE_Y + 20), (0, 255, 0))
        self.draw_string(self.info_font, 'Action            Confidence', (self.AP_TITLE_X, self.AP_TITLE_Y + 40), (0, 255, 255))

        if self.action_probabilities is None:
            return

        y = self.AP_TITLE_Y + 60
        for button in self.button_names:
            self.draw_string(self.font, button, (self.AP_X, y), (255, 255, 255))
            y += 30

        y = self.AP_TITLE_Y + 60
        for prob in self.action_probabilities:
            self.draw_string(self.font, ('%f' % prob), (self.AP_X + 150, y), (255, 255, 255))
            y += 30

        # self.draw_string(self.info_font, '84x84 pixels', (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 20), (0, 255, 255))
        # self.draw_string(self.info_font, 'last 4 frames', (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 40), (0, 255, 255))

        # img = np.array(input_state[0])

        # frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # surf = pygame.surfarray.make_surface(frame)
        # self.main_surf.blit(pygame.transform.rotozoom(surf, -90, 3), (self.INPUT_X, self.INPUT_Y))

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

    def set_ai_sys_info(self, ai_sys):

        self.action_probabilities = ai_sys.display_probs
        
        if ai_sys:
            self.model_1_num_params = ai_sys.model_1_num_params
            self.model_2_num_params = ai_sys.model_2_num_params
            self.model_in_use = ai_sys.model_in_use

            

            if self.model_in_use == 1:
                self.model_name = "DEFENSE ZONE"
            elif self.model_in_use == 2:
                self.model_name = "SCORING OPP"
            else:
                self.model_name = "N/A"

    def DrawFrameRewardHistogram(self, posX, posY, width, height):
        

        #plt.plot(self.frameRewardList, color=(0,1,0))

        #self.frameRewardList = [1] * 200
        #plt.plot(self.frameRewardList, color=(0,1,0))
        #plt.hist(self.frameRewardList, bins=1, rwidth=0.8)
        #plt.bar(self.frameRewardList, 50)

        #plt.title(("Reward Mean: %d" % broadcast.rewardmean), color=(0,1,0))

        


        #draw buffer
        #self.fig.canvas.draw()
        width, height = self.fig.canvas.get_width_height()
        buffer, size = self.fig.canvas.print_to_buffer()
        #print(buffer)
        image = np.fromstring(buffer, dtype='uint8').reshape(height, width, 4)
        #self.final[posY:posY+height,posX:posX+width] = image[:,:,0:3]

        #print(image[:,:,0:3])
        #rf_img = np.transpose(image[:,:,0:3], (1,0,2))

        #surf = pygame.surfarray.make_surface(rf_img)
        #surf.fill((1, 1, 1))
        #self.screen.blit(surf, (1280, 600))

        #plt.close()

        self.draw_string(self.info_font, 'REWARD FUNCTION', (1280, 600), (0, 255, 0))

        self.frameListUpdated = False
                

    def draw_frame(self, frame_img, action_probabilities, input_state, info):
        self.screen.fill((30, 30, 30))
        emu_screen = np.transpose(frame_img, (1,0,2))

        surf = pygame.surfarray.make_surface(emu_screen)

        self.screen.blit(pygame.transform.scale(surf,(self.GAME_WIDTH,self.GAME_HEIGHT)), (0, 0))

        
        self.draw_contact_info()
        self.draw_basic_info()
        self.draw_model(input_state)
        #self.draw_action_probabilties(self.action_probabilities)
        #self.DrawFrameRewardHistogram(self.GAME_WIDTH,400,100,20)

        self.draw_ai_overview()

        pygame.display.flip()

        self.get_input()

        keystate = self.get_input()
        self.ProcessKeyState(keystate)
        

    def ProcessKeyState(self, keystate):

        if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
            exit()

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

    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()

        return keystate
