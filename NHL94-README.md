# NHL94


## Examples

Models (Player 1) vs in-game AI (Player 2):
```bash
python3 model_vs_game.py --env=NHL941on1-Genesis --state=PenguinsVsSenators --model_1=./models/DefenseZone --model_2=./models/ScoreGoal --nn=MlpPolicy --rf=General
```
Play against the models
```bash
python3 player_vs_model.py --env=NHL941on1-Genesis --state=PenguinsVsSenators.2P --model_1=./models/DefenseZone --model_2=./models/ScoreGoal --nn=MlpPolicy --rf=General --num_players=2
```

Play against a model with for specific reward function (ex: DefenseZone):
```bash
python3 player_vs_model.py --env=NHL941on1-Genesis --state=PenguinsVsSenators.lostpuck.2P --nn=MlpPolicy --num_players=2 --rf="DefenseZone" --load_model=./models/DefenseZone
```

Train a model for a specific reward function (ex: ScoreGoal):
```bash
python3 model_trainer.py --env=NHL941on1-Genesis --state=PenguinsVsSenators.FrontOfNet --num_env=12 --num_timesteps=100_000_000 --nn=MlpPolicy --play --num_players=1 --rf="ScoreGoal"
```

## Devlog

For NHL94:
https://youtube.com/playlist?list=PLmwlWbdWpZVtH6NXqWbrnWOf6SWv9nJBY

NHL94 Discord (with a subgroup dedicated for AI):
https://discord.gg/SDnKEXujDs
