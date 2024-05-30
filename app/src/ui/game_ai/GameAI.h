#pragma once

#include <bitset>
#include <string>
#include <filesystem>
#include "data.h"

class RetroModel {
public:
        virtual void Forward(std::vector<float> & output, const std::vector<float> & input)=0;

};

class GameAI {
public:
        virtual void Init(std::filesystem::path dir) {};
        RetroModel * LoadModel(std::string path);

        virtual void Think(std::bitset<16> & buttons, const Retro::GameData & m_data) {};

        static GameAI * CreateGameAI(std::string name);
};