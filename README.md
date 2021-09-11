# Model vs Model

Train and pit two models against each other on PvP retro games such as Mortal Kombat or WWF Wrestlemania: The Arcade Game

## Install dependencies

Requires:
*   Tensorflow 1.X, 1.14 recommended
*   stable-baselines 2.10 (fork of baselines)
*   stable-retro (fork of gym-retro)

```
pip3 install opencv-python anyrl gym joblib atari-py tensorflow-gpu==1.14 stable-baselines stable-retro pygame
```

## Install roms
You need to provide your own roms

In your rom directory exec this command, it will import the roms into stable-retro
```
python3 -m retro.import .
```

If your game is not integrated you need to use OpenAI's integration tool to specify the reward function and other info useful for the algo:

[More details can be found here on how to import existing games or integrate new ones](https://www.videogames.ai/2019/01/29/Setup-OpenAI-baselines-retro.html)

## Example Usage

### Train your Models

Train Yokozuna Model:
```
python3 trainer.py --env=WWFArcade-Genesis --state=VeryHard_YokozunaVsShawnMicheals --num_timesteps=5000000 --play
```

Train Shawn Micheals Model:
```
python3 trainer.py --env=WWFArcade-Genesis --state=VeryHard_ShawnMichealsVsBrettHart.state --num_timesteps=5000000 --play
```

The models (zip files) should reside in the output directory (by default in the /home directory)


### Pit your two models against each other
```
python3 multiplayer.py --env=WWFArcade-Genesis --load_p1_model=~/yokozuna.zip --load_p2_model=~/shawn_micheals.zip
```

### Game specific training script to beat WWF (Continental mode)
```
python3 wwf_trainer.py --play
```


### Play pre-trained model
```
python3 singleplayer.py --env=WWFArcade-Genesis --state=VeryHard_YokozunaVsShawnMicheals --load_p1_model=~/yokozuna.zip
```