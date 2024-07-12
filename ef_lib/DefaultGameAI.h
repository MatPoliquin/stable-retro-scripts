#pragma once

#include "GameAI.h"
#include "memory.h"
#include "data.h"



class DefaultGameAI : public GameAI {
public:
    virtual void Init(const char * dir, void * ram_ptr, int ram_size);

    virtual void Think(bool buttons[GAMEAI_MAX_BUTTONS], int player, const void *frame_data, unsigned int frame_width, unsigned int frame_height, unsigned int frame_pitch);


private:
    RetroModel * model;

    Retro::GameData retro_data;
};