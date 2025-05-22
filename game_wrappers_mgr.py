from game_wrappers.nhl94_obs import NHL94Observation2PEnv
from game_wrappers.nhl94_display import NHL94GameDisplayEnv
from game_wrappers.nhl94_display_pvp import NHL94PvPGameDisplayEnv
from game_wrappers.nhl94_ai import NHL94AISystem

from game_wrappers.pong_obs import PongObservationEnv
from game_wrappers.display import PvPGameDisplayEnv, GameDisplayEnv
from game_wrappers.ai_sys import AISys
from game_wrappers.compare_model_display import CompareModelDisplay

class GameWrapperManager(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GameWrapperManager, cls).__new__(cls)
        return cls.instance
    def __init__(self):
        self.obs_env = None
        self.pvp_display_env = None
        self.sp_display_env = None
        self.ai_sys = None
        self.compare_model = None

    def init(self, args):
        self.obs_env = CompareModelDisplay
        self.pvp_display_env = PvPGameDisplayEnv
        self.sp_display_env = GameDisplayEnv
        self.ai_sys = AISys
        self.compare_model = CompareModelDisplay

        # overide with game specific wrappers
        if args.env == 'NHL941on1-Genesis' or args.env == 'NHL942on2-Genesis' or args.env == 'NHL94-Genesis':
            self.obs_env = NHL94Observation2PEnv
            self.pvp_display_env = NHL94PvPGameDisplayEnv
            self.sp_display_env = NHL94GameDisplayEnv
            self.ai_sys = NHL94AISystem
        elif args.env == 'Pong-Atari2600':
            self.obs_env = PongObservationEnv

wrappers = GameWrapperManager()