# NHL94


## Examples
Note: Currently there is two NHL 94 env in stable-retro: The original game and the '1 on 1' rom hack which as the name implies is for 1 on 1 matches instead of full teams

Models (Player 1) vs in-game AI (Player 2):
```bash
python3 play.py --mode=model_vs_game --env=NHL941on1-Genesis --state=PenguinsVsSenators --model_1=./models/DefenseZone --model_2=../models/ScoreGoal --nn=MlpPolicy --rf=General_1P
```
Play against the models
```bash
python3 play.py --mode=player_vs_model --env=NHL941on1-Genesis --state=PenguinsVsSenators.2P --model_1=./models/DefenseZone --model_2=../models/ScoreGoal --nn=MlpPolicy --rf=General_1P --num_players=2
```

### Play against a model with for specific reward function
Note that DefenseZone is expected to be on model_1 slot and ScoreGoal is expected to be model_2 slot
ex: DefenseZone
```bash
python3 play.py --mode=player_vs_model --env=NHL941on1-Genesis --state=PenguinsVsSenators.lostpuck.2P --nn=MlpPolicy --num_players=2 --rf="DefenseZone_1P" --model_1=../models/DefenseZone
```
ex: ScoreGoal
```bash
python3 play.py --mode=player_vs_model --env=NHL941on1-Genesis --state=PenguinsVsSenators.frontnet.2P --nn=MlpPolicy --num_players=2 --rf="ScoreGoal_1P" --model_2=../models/ScoreGoal
```


### Train a model for a specific reward function (ex: ScoreGoal):
```bash
python3 train.py --env=NHL941on1-Genesis --state=PenguinsVsSenators.FrontOfNet --num_env=12 --num_timesteps=100_000_000 --nn=MlpPolicy --play --num_players=1 --rf="ScoreGoal_1P" --hyperparams=../hyperparams/nhl94.json
```

### Play against the game
Useful for debugging purposes, check /scripts/game_wrappers/nhl94_display_debug.py
```bash
python3 play.py --mode=player_vs_game --env=NHL94-Genesis --num_env=1 --nn=MlpPolicy --num_players=1 --rf="General" --hyperparams=../hyperparams/nhl94.json
```

## Devlog

NHL94 Discord (with a subgroup dedicated for AI):
https://discord.gg/SDnKEXujDs

Video of AI playing NHL94 (1 on 1) with explaination about the reward functions:
https://www.youtube.com/watch?v=UBXXn2amGUU
