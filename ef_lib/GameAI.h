#pragma once

#include <bitset>
#include <string>
#include <filesystem>
#include <vector>
//#include "data.h"

class RetroModel {
public:
        virtual void Forward(std::vector<float> & output, const std::vector<float> & input)=0;

};

class GameAI {
public:
        virtual void Init(std::filesystem::path dir, void * ram_ptr, int ram_size) {};
        RetroModel * LoadModel(std::string path);

        virtual void Think(std::bitset<16> & buttons) {};

        
};


typedef GameAI * (*creategameai_t)(std::string);
typedef int (*testfunc_t)();

//extern "C" GameAI * CreateGameAI(std::string name);