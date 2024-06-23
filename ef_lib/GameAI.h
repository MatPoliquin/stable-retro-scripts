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


#define GAMEAI_MAX_BUTTONS 16

class GameAI {
public:
        virtual void Init(const char * dir, void * ram_ptr, int ram_size) {};
        RetroModel * LoadModel(const char * path);

        virtual void Think(bool buttons[GAMEAI_MAX_BUTTONS]) {};

        
};


typedef GameAI * (*creategameai_t)(const char *);
typedef int (*testfunc_t)();

//extern "C" GameAI * CreateGameAI(std::string name);