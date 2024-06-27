# stable-retro lib
Library to be used with emulator frontends (such as RetroArch) to enable ML models to overide player input.
Warning: Still in early prototype version

## Building the lib for linux

```
sudo apt update
sudo apt install git cmake unzip libqt5opengl5-dev qtbase5-dev zlib1g-dev python3 python3-pip build-essential
```

```
git clone https://github.com/MatPoliquin/stable-retro-scripts.git
```

Download pytorch C++ lib:
```
cd stable-retro-scripts/ef_lib/
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.3.1%2Bcpu.zip
unzip libtorch-shared-with-deps-2.3.1+cpu.zip
```

Generate makefiles and compile
```
cmake . -DCMAKE_PREFIX_PATH=./libtorch
make
```

## Test the lib
You can test this dynamic lib with the prototype app in /app

If you want to use it with RetroArch I added support of the lib in this fork:
https://github.com/MatPoliquin/RetroArchML
