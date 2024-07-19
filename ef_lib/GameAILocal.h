#pragma once

#include "GameAI.h"
#include "RetroModel.h"

#include <bitset>
#include <string>
#include <filesystem>
#include <vector>
#include <queue>
//#include "data.h"




class GameAILocal : public GameAI {
public:
        GameAILocal():showDebug(false),
                debugLogFunc(nullptr){};


        RetroModel * LoadModel(const char * path);
        
        void SetShowDebug(const bool show){ this->showDebug = show; };

        void SetDebugLog(debug_log_t func){debugLogFunc = func;};

protected:
        void DebugPrint(const char * msg);

        bool            showDebug;
        debug_log_t     debugLogFunc;

public:
        std::string full_path;
        std::string dir_path;
        std::string game_name;
};


