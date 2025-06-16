# Wrestlemania the arcade game


Train Yokozuna Model:
```
python3 train.py --env=WWFArcade-Genesis --state=VeryHard_YokozunaVsShawnMicheals --num_timesteps=5000000 --play
```

Train Shawn Micheals Model:
```
python3 train.py --env=WWFArcade-Genesis --state=VeryHard_ShawnMichealsVsYokozuna --num_timesteps=5000000 --play
```

The models (zip files) should reside in the output directory (by default in the /home directory)


### Pit your two models against each other
```
python3 play.py --mode=model_vs_model --env=WWFArcade-Genesis --load_p1_model=~/yokozuna.zip --load_p2_model=~/shawn_micheals.zip
```

### Game specific training script to beat WWF (Continental mode)
```
python3 -m custom_trainers.wwf_trainer --play
```

### Play pre-trained model
```
python3 play.py --mode=model_vs_game --env=WWFArcade-Genesis --state=VeryHard_YokozunaVsShawnMicheals --model_1=~/yokozuna.zip
```
