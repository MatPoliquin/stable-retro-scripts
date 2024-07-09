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

typedef void (*debug_log_t)(int level, const char *fmt, ...);


#define GAMEAI_MAX_BUTTONS 16

class GameAI {
public:
        GameAI():showDebug(false),
                debugLogFunc(nullptr){};

        virtual void Init(const char * dir, void * ram_ptr, int ram_size) {};
        RetroModel * LoadModel(const char * path);
        
        virtual void Think(bool buttons[GAMEAI_MAX_BUTTONS], int player=0) {};

        void SetShowDebug(const bool show){ this->showDebug = show; };

        void SetDebugLog(debug_log_t func){debugLogFunc = func;};

protected:
        void DebugPrint(const char * msg);

        bool            showDebug;
        debug_log_t     debugLogFunc;
};


typedef GameAI * (*creategameai_t)(const char *);
typedef int (*testfunc_t)();