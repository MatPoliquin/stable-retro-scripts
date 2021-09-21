# Retro Scripts

* Train models on retro games
* Pit two models against each other on PvP retro games such as Mortal Kombat or WWF Wrestlemania: The Arcade Game
* Pygame based display to make videos

## Install dependencies

Requires:
*   Tensorflow 1.X, 1.14 recommended
*   stable-baselines 2.10 (fork of baselines)
*   stable-retro (fork of gym-retro)

```
sudo apt-get update
sudo apt-get install cmake python3 python3-pip git zlib1g-dev libopenmpi-dev ffmpeg
```

```
pip3 install opencv-python anyrl gym joblib atari-py tensorflow-gpu==1.14 baselines stable-baselines pygame

git clone https://github.com/MatPoliquin/stable-retro.git
cd stable-retro
pip3 install -e .
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
python3 model_trainer.py --env=WWFArcade-Genesis --state=VeryHard_YokozunaVsShawnMicheals --num_timesteps=5000000 --play
```

Train Shawn Micheals Model:
```
python3 model_trainer.py --env=WWFArcade-Genesis --state=VeryHard_ShawnMichealsVsYokozuna --num_timesteps=5000000 --play
```

The models (zip files) should reside in the output directory (by default in the /home directory)


### Pit your two models against each other
```
python3 model_vs_model.py --env=WWFArcade-Genesis --load_p1_model=~/yokozuna.zip --load_p2_model=~/shawn_micheals.zip
```

### Game specific training script to beat WWF (Continental mode)
```
python3 wwf_trainer.py --play
```


### Play pre-trained model
```
python3 model_vs_game.py --env=WWFArcade-Genesis --state=VeryHard_YokozunaVsShawnMicheals --load_p1_model=~/yokozuna.zip
```
