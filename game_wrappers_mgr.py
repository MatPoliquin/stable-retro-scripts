from game_wrappers.nhl94_obs import NHL94ObservationEnv, NHL94Discretizer
from game_wrappers.nhl94_display import NHL94PvPGameDisplayEnv, NHL94GameDisplayEnv
from game_wrappers.pong_obs import PongObservationEnv
from game_wrappers.display import PvPGameDisplayEnv, GameDisplayEnv
from game_wrappers.nhl94_ai import NHL94AISystem
from game_wrappers.ai_sys import AISys

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

    def init(self, args):
        if args.env == 'NHL941on1-Genesis':
            self.obs_env = NHL94ObservationEnv
            self.pvp_display_env = NHL94PvPGameDisplayEnv
            self.sp_display_env = GameDisplayEnv
            self.ai_sys = NHL94AISystem
            print("========= NHL941on1-Genesis")
            print(self.obs_env)
        elif args.env == 'Pong-Atari2600':
            
            self.obs_env = PongObservationEnv
            self.pvp_display_env = PvPGameDisplayEnv
            self.sp_display_env = GameDisplayEnv
            self.ai_sys = AISys
            print("=========== Pong-Atari2600")
            print(self.obs_env)
        else:
            self.obs_env = None
            self.pvp_display_env = PvPGameDisplayEnv
            self.sp_display_env = GameDisplayEnv
            self.ai_sys = AISys
    

wrappers = GameWrapperManager()