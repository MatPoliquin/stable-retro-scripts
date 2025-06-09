"""
Display PvP
"""

from os import environ
import gymnasium as gym
import numpy as np
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame # pylint: disable=wrong-import-position,wrong-import-order
import pygame.freetype # pylint: disable=wrong-import-position,wrong-import-order

class NHL94PvPGameDisplayEnv():
    def __init__(self, env, args, model1_desc, model2_desc, model1_params, model2_params, button_names):
        self.env = env

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

        framebuffer = self.env.render()

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
