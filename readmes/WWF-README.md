# Wrestlemania the arcade game

All command examples below assume you run them from the repository root.

Train Yokozuna Model:
```
python3 scripts/train.py --env=WWFArcade-Genesis --state=VeryHard_YokozunaVsShawnMicheals --num_timesteps=5000000 --play
```

Train Shawn Micheals Model:
```
python3 scripts/train.py --env=WWFArcade-Genesis --state=VeryHard_ShawnMichealsVsYokozuna --num_timesteps=5000000 --play
```

The models (zip files) should reside in the output directory (by default in the /home directory)


### Pit your two models against each other
```
python3 scripts/play.py --mode=model_vs_model --env=WWFArcade-Genesis --load_p1_model=~/yokozuna.zip --load_p2_model=~/shawn_micheals.zip
```

### Play pre-trained model
```
python3 scripts/play.py --mode=model_vs_game --env=WWFArcade-Genesis --state=VeryHard_YokozunaVsShawnMicheals --model_1=~/yokozuna.zip
```
