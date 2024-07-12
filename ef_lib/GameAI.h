#pragma once

#include <bitset>
#include <string>
#include <filesystem>
#include <vector>
//#include "data.h"


class RetroModelFrameData
{
public:
        void *data;
        unsigned int width;
        unsigned int height;
        unsigned int pitch;
};

class RetroModel {
public:
        virtual void Forward(std::vector<float> & output, const std::vector<float> & input)=0;
        virtual void Forward(std::vector<float> & output, RetroModelFrameData & input)=0;
};

typedef void (*debug_log_t)(int level, const char *fmt, ...);


#define GAMEAI_MAX_BUTTONS 16

class GameAI {
public:
        GameAI():showDebug(false),
                debugLogFunc(nullptr){};

        virtual void Init(const char * dir, void * ram_ptr, int ram_size) {};
        RetroModel * LoadModel(const char * path);
        
        virtual void Think(bool buttons[GAMEAI_MAX_BUTTONS], int player, const void *frame_data, unsigned int frame_width, unsigned int frame_height, unsigned int frame_pitch) {};

        void SetShowDebug(const bool show){ this->showDebug = show; };

        void SetDebugLog(debug_log_t func){debugLogFunc = func;};

protected:
        void DebugPrint(const char * msg);

        bool            showDebug;
        debug_log_t     debugLogFunc;
};


typedef GameAI * (*creategameai_t)(const char *);
typedef int (*testfunc_t)();