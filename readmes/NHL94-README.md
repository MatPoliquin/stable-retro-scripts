# NHL94


## Examples

Models (Player 1) vs in-game AI (Player 2):
```bash
python3 model_vs_game.py --env=NHL941on1-Genesis --state=PenguinsVsSenators --model_1=./models/DefenseZone --model_2=../models/ScoreGoal --nn=MlpPolicy --rf=General_1P
```
Play against the models
```bash
python3 player_vs_model.py --env=NHL941on1-Genesis --state=PenguinsVsSenators.2P --model_1=./models/DefenseZone --model_2=../models/ScoreGoal --nn=MlpPolicy --rf=General_1P --num_players=2
```

### Play against a model with for specific reward function
Note that DefenseZone is expected to be on model_1 slot and ScoreGoal is expected to be model_2 slot
ex: DefenseZone
```bash
python3 player_vs_model.py --env=NHL941on1-Genesis --state=PenguinsVsSenators.lostpuck.2P --nn=MlpPolicy --num_players=2 --rf="DefenseZone_1P" --model_1=../models/DefenseZone
```
ex: ScoreGoal
```bash
python3 player_vs_model.py --env=NHL941on1-Genesis --state=PenguinsVsSenators.frontnet.2P --nn=MlpPolicy --num_players=2 --rf="ScoreGoal_1P" --model_2=../models/ScoreGoal
```


### Train a model for a specific reward function (ex: ScoreGoal):
```bash
python3 model_trainer.py --env=NHL941on1-Genesis --state=PenguinsVsSenators.FrontOfNet --num_env=12 --num_timesteps=100_000_000 --nn=MlpPolicy --play --num_players=1 --rf="ScoreGoal_1P" --hyperparams=../hyperparams/nhl94.json
```

## Devlog

NHL94 Discord (with a subgroup dedicated for AI):
https://discord.gg/SDnKEXujDs

Video of AI playing NHL94 (1 on 1) with explaination about the reward functions:
https://www.youtube.com/watch?v=UBXXn2amGUU
