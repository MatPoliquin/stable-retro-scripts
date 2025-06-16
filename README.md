![pylint workflow](https://github.com/MatPoliquin/stable-retro-scripts/actions/workflows/pylint.yml/badge.svg)
![clang workflow](https://github.com/MatPoliquin/stable-retro-scripts/actions/workflows/clang.yml/badge.svg)
![testcpp workflow](https://github.com/MatPoliquin/stable-retro-scripts/actions/workflows/test-cpp.yml/badge.svg)

# stable-retro scripts

* Train models on retro games
* Pit two models against each other on PvP retro games such as NHL94, Mortal Kombat or WWF Wrestlemania: The Arcade Game
* Play against an improved AI opponent
* Emulator Frontend library using Pytorch C++ to able to play with or against stable-retro models in apps like RetroArch

### Supported models
*   MLPs
*   Nature CNN (from DeepMind)
*   Impala CNN (from DeepMind)
*   Combined Input models (image + scalar)

Experimental:
*   Attention MLPs

NHL94 (1 on 1)           |  Wrestlemania: The Arcade game |  Virtua Fighter 1
:-------------------------:|:-------------------------:|:-------------------------:
![screenshot 01](./screenshots/nhl94.png)  |  ![wwf vs](./screenshots/wwf.png) | ![vf](./screenshots/virtua_fighter.png)


## Installation

Tested on Ubuntu 22.04/24.04 and Windows 11 WSL2 (Ubuntu 22.04 VM)

Requires:
*   Python 3.7 and up
*   gymnasium
*   stable-baselines3
*   stable-retro (fork of gym-retro)

```
sudo apt update
sudo apt-get install python3 python3-pip git zlib1g-dev libopenmpi-dev ffmpeg cmake

sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ~/vretro
source ~/vretro/bin/activate

git clone https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro
pip3 install -e .

pip3 install "stable_baselines3[extra]" pygame torchsummary opencv-python timm
```

Windows WSL2 + Ubuntu 22.04 setup guide: https://www.youtube.com/watch?v=vPnJiUR21Og

## Install roms
You need to provide your own roms

In your rom directory exec this command, it will import the roms into stable-retro
```
python3 -m retro.import .
```

### Train a model
Note: Airstriker is a public domain rom and is already included in stable-retro
```bash
python3 train.py --env=Airstriker-Genesis --nn=CnnPolicy --num_env=8 --num_timesteps=1_000_000 --play --hyperparams=../hyperparams/default.json
```

## Game specific Examples

*   For NHL94 specific page click [here](./readmes/NHL94-README.md)
*   For Wrestlemania the arcade game specific page click [here](./readmes/WWF-README.md)

## retro ai lib

C++ lib using Pytorch that runs models inside emulator frontends like retro arch to override player input. Which means you can play against a smarter opponent at NHL94 for example or let the AI play with you in COOP or play for you.

See [README](./retro_ai_lib/README.md) for build and install instructions

Tutorial video:
[![RetroArch and Pytorch](https://img.youtube.com/vi/hkOcxJvJVjk/0.jpg)](https://www.youtube.com/watch?v=hkOcxJvJVjk)
