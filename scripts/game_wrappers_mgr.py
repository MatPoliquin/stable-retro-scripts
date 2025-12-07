from game_wrappers.nhl94_obs import NHL94Observation2PEnv
from game_wrappers.nhl94_display import NHL94GameDisplayEnv
from game_wrappers.nhl94_display_debug import NHL94DebugDisplay
from game_wrappers.nhl94_display_pvp import NHL94PvPGameDisplayEnv
from game_wrappers.nhl94_ai import NHL94AISystem

from game_wrappers.pong_obs import PongObservationEnv, PongTemporalObservationEnv
from game_wrappers.fighter_obs import FighterObservationEnv
from game_wrappers.display import GameDisplayEnv
from game_wrappers.display_pvp import PvPGameDisplayEnv
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
        self.pvp_display_env = PvPGameDisplayEnv
        self.sp_display_env = GameDisplayEnv
        self.ai_sys = AISys

        # overide with game specific wrappers
        if args.env in ('NHL941on1-Genesis-v0', 'NHL942on2-Genesis-v0', 'NHL94-Genesis-v0'):
            self.obs_env = NHL94Observation2PEnv
            self.pvp_display_env = NHL94PvPGameDisplayEnv
            self.sp_display_env = NHL94DebugDisplay
            self.ai_sys = NHL94AISystem
        elif args.env == 'Pong-Atari2600-v0':
            if args.nn == 'EntityAttentionPolicy':
                self.obs_env = PongTemporalObservationEnv
            else:
                self.obs_env = PongObservationEnv
        elif args.env == 'MortalKombatII-Genesis-v0':
            self.obs_env = FighterObservationEnv

wrappers = GameWrapperManager()