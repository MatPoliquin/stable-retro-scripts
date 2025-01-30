"""
Display
"""

from os import environ
import gymnasium as gym
import numpy as np
from game_wrappers.nhl941on1_rf import rf_defensezone, rf_scoregoal
from game_wrappers.nhl941on1_gamestate import NHL941on1GameState
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame # pylint: disable=wrong-import-position,wrong-import-order
import pygame.freetype # pylint: disable=wrong-import-position,wrong-import-order


class NHL941on1GameDisplayEnv(gym.Wrapper):
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
        self.STATS_X = 600
        self.STATS_Y = self.GAME_HEIGHT + 10
        self.RF_X = self.GAME_WIDTH + 10
        self.RF_Y = 700

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

        self.scr_count = 0

        self.action_probabilities = None
        self.player_actions = [0] * 12

        self.frameRewardList = [0.0] * 200

        self.game_state = NHL941on1GameState()

        self.model_in_use = 0
        self.model_params = [None, None, None]
        self.model_names = ["CODE", "DEFENSE ZONE", "SCORING OPP"]

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def set_reward(self, rew):
        #self.frameListUpdated = True
        #return
        #print(rew)
        self.reward = rew
        self.frameRewardList.append(rew)
        self.frameRewardList = self.frameRewardList[1:len(self.frameRewardList)]
        #if rew != 0:
        #    print(rew)

        #self.frameListUpdated = True

    def step(self, ac):
        ob, rew, done, info = self.env.step(ac)

        # get rewards for current model
        rew = 0
        self.game_state.BeginFrame(info[0])
        if self.model_in_use == 1:
            rew = rf_defensezone(self.game_state)
        elif self.model_in_use == 2:
            rew = rf_scoregoal(self.game_state)

        self.game_state.EndFrame()

        self.set_reward(rew)

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
        self.draw_string(self.info_font, 'GAME STATS', (self.MODELDESC_X, self.MODELDESC_Y), (0, 255, 0))
        #self.draw_string(self.info_font, 'NUM PARAMETERS', (self.NUM_PARAMS_X, self.NUM_PARAMS_Y), (0, 255, 0))

        self.draw_string(self.font, self.args.env, (self.ENV_X - 100, self.ENV_Y - 70), (255, 255, 255))
        #self.draw_string(self.info_font_big, '2xMLP', (self.MODELDESC_X, self.MODELDESC_Y - 70), (255, 255, 255))
        #self.draw_string(self.info_font_big, ('%d' % self.num_params), (self.NUM_PARAMS_X, self.NUM_PARAMS_Y - 70), (255, 255, 255))

    def draw_ai_overview(self):

        pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(self.AI_X - 5, self.AI_Y - 5, 580, 100), width=3)

        self.draw_string(self.info_font, 'MODEL(TYPE): NUM PARAMS', (self.AI_X, self.AI_Y), (0, 255, 0))

        i = 0
        for p in self.model_params:
            name = self.model_names[i]
            num_params = 'N/A' if p == None else p
            color = (255, 255, 255)
            if self.model_in_use == i:
                color = (0, 0, 255)

            self.draw_string(self.info_font, ('%s: %s' % (name, num_params)), (self.AI_X, self.AI_Y + 20 + i * 20), color)
            i += 1

    def draw_model(self, input_state):
        self.draw_string(self.info_font, ('CURRENT MODEL:%s' % self.model_names[self.model_in_use]), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y - 25), (0, 255, 0))
        pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(self.INPUT_TITLE_X - 5, self.INPUT_TITLE_Y - 5, 580, 470), width=3)


        self.draw_string(self.info_font, 'INPUT', (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 20), (0, 255, 0))

        color = (255, 255, 255)
        if self.model_in_use == 0:
            color = (50, 50, 50)

        self.draw_string(self.info_font, ('P1 POS: (%.3f, %.3f)' % (input_state[0][0], input_state[0][1])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 40), color)
        self.draw_string(self.info_font, ('P1 VEL: (%.3f, %.3f)' % (input_state[0][2], input_state[0][3])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 60), color)
        self.draw_string(self.info_font, ('P2 POS: (%.3f, %.3f)' % (input_state[0][4], input_state[0][5])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 80), color)
        self.draw_string(self.info_font, ('P2 VEL: (%.3f, %.3f)' % (input_state[0][6], input_state[0][7])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 100), color)
        self.draw_string(self.info_font, ('PUCK POS: (%.3f, %.3f)' % (input_state[0][8], input_state[0][9])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 120), color)
        self.draw_string(self.info_font, ('PUCK VEL: (%.3f, %.3f)' % (input_state[0][10], input_state[0][11])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 140), color)
        self.draw_string(self.info_font, ('G2 POS: (%.3f, %.3f)' % (input_state[0][12], input_state[0][13])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 160), color)
        self.draw_string(self.info_font, ('P1/G1 HASPUCK: (%.1f, %.1f)' % (input_state[0][14], input_state[0][15])), (self.INPUT_TITLE_X, self.INPUT_TITLE_Y + 180), color)

        self.draw_string(self.info_font, 'OUTPUT', (self.AP_TITLE_X, self.AP_TITLE_Y + 20), (0, 255, 0))
        self.draw_string(self.info_font, 'Action            Confidence', (self.AP_TITLE_X, self.AP_TITLE_Y + 40), (0, 255, 255))

        if self.action_probabilities is None:
            return

        y = self.AP_TITLE_Y + 60
        for button in self.button_names:
            self.draw_string(self.font, button, (self.AP_X, y), color)
            y += 30

        y = self.AP_TITLE_Y + 60
        for prob in self.action_probabilities:
            self.draw_string(self.font, ('%f' % prob), (self.AP_X + 150, y), color)
            y += 30

    def draw_game_stats(self, info):
        #self.draw_string(self.info_font, 'GAME STATS', (self.STATS_X, self.STATS_Y), (0, 255, 0))

        self.draw_string(self.info_font, ('P1 SHOTS: %d' %  info.get('p1_shots')), (self.STATS_X, self.STATS_Y), (0, 255, 255))
        self.draw_string(self.info_font, ('P2 SHOTS: %d' %  info.get('p2_shots')), (self.STATS_X + 300, self.STATS_Y), (0, 255, 255))

        self.draw_string(self.info_font, ('P1 ATKZONE: %d' %  info.get('p1_attackzone')), (self.STATS_X, self.STATS_Y + 20), (0, 255, 255))
        self.draw_string(self.info_font, ('P2 ATKZONE: %d' %  info.get('p2_attackzone')), (self.STATS_X + 300, self.STATS_Y + 20), (0, 255, 255))

        self.draw_string(self.info_font, ('P1 BODYCHECKS: %d' %  info.get('p1_passing')), (self.STATS_X, self.STATS_Y + 40), (0, 255, 255))
        self.draw_string(self.info_font, ('P2 BODYCHECKS: %d' %  info.get('p2_passing')), (self.STATS_X + 300, self.STATS_Y + 40), (0, 255, 255))

        self.draw_string(self.info_font, ('P1 FACEOFFWON: %d' %  info.get('p1_faceoffwon')), (self.STATS_X, self.STATS_Y + 60), (0, 255, 255))
        self.draw_string(self.info_font, ('P2 FACEOFFWON: %d' %  info.get('p2_faceoffwon')), (self.STATS_X + 300, self.STATS_Y + 60), (0, 255, 255))


    def set_ai_sys_info(self, ai_sys):

        self.action_probabilities = ai_sys.display_probs

        if ai_sys:
            self.model_params = ai_sys.model_params
            self.model_in_use = ai_sys.model_in_use

        #print(self.model_in_use)

    def DrawFrameRewardHistogram(self, posX, posY, width, height):

        self.draw_string(self.info_font, ('REWARD FUNCTION:%f' % self.frameRewardList[-1]), (self.RF_X, self.RF_Y - 5), (0, 0, 255))


        bar_height = 50

        pygame.draw.line(self.screen, (255,255,255), (self.RF_X, self.RF_Y + 100), (self.RF_X + 600, self.RF_Y + 100))
        pygame.draw.line(self.screen, (255,255,255), (self.RF_X, self.RF_Y + 100 - bar_height), (self.RF_X, self.RF_Y + 100 + bar_height))

        i = 0

        for r in self.frameRewardList:
            i += 3
            height = abs(r) * bar_height
            y = (self.RF_Y + 100) if r < 0.0 else (self.RF_Y + 100 - height)
            if r != 0:
                pygame.draw.rect(self.screen, (0,255,0), pygame.Rect(self.RF_X + i, y, 2, height))


    def draw_frame(self, frame_img, action_probabilities, input_state, info):

        self.screen.fill((30, 30, 30))

        # Draw game frame
        emu_screen = np.transpose(frame_img, (1,0,2))
        surf = pygame.surfarray.make_surface(emu_screen)
        self.screen.blit(pygame.transform.scale(surf,(self.GAME_WIDTH,self.GAME_HEIGHT)), (0, 0))

        # Draw info
        self.draw_contact_info()
        self.draw_basic_info()
        self.draw_model(input_state)
        self.DrawFrameRewardHistogram(self.GAME_WIDTH,400,100,20)
        self.draw_game_stats(info[0])
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


        if keystate[pygame.K_i]:
            self.scr_count += 1
            pygame.image.save(self.screen, f"screenshot0{self.scr_count}.png")


    def get_input(self):
        pygame.event.pump()
        keystate = pygame.key.get_pressed()

        return keystate
