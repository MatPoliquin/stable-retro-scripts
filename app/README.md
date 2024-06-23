# stable-retro play
Emulator frontend that supports Machine Learning models to control players. The source code is based on OpenAI's gym-retro-integration tool.

This is a prototype version so only NHL94 (1 on 1 Sega Genesis version) is supported for now.

## Building the app

```
sudo apt update
sudo apt install git cmake unzip libqt5opengl5-dev qtbase5-dev zlib1g-dev python3 python3-pip build-essential
```

```
git clone https://github.com/MatPoliquin/stable-retro-scripts.git
```

Generate makefiles and compile
```
cd app
cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY ..
make
```

## Example
Launch the app
```
./sr-play
```

Load NHL94 1on1
*   On the top right menu, click: Game->Load Game
*   Open your rom file in app/retro/data/NHL941on1-Genesis

Load 2 player state
*   On the top right menu, click: Game->Load State
*   Open the following state: app/retro/data/NHL941on1-Genesis/PenguinsVsSenators.2P.state

You should now be able to play againsts the custom AI. You control player 2 and the AI controls player 1.
You might need to config the inputs using Windows->Controls menu
