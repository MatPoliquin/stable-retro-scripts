#pragma once

#include "GameAI.h"
#include "memory.h"
#include "data.h"

class NHL94Data;

class NHL94GameAI : public GameAI {
public:
    virtual void Init(const char * dir, void * ram_ptr, int ram_size);

    void SetModelInputs(std::vector<float> & input, const NHL94Data & data);
    virtual void Think(bool buttons[GAMEAI_MAX_BUTTONS], int player, const void *frame_data, unsigned int frame_width, unsigned int frame_height, unsigned int frame_pitch, unsigned int pixel_format);
    void GotoTarget(std::vector<float> & input, int vec_x, int vec_y);

private:
    RetroModel * ScoreGoalModel;
    RetroModel * DefenseModel;
    bool isShooting;

    Retro::GameData retro_data;
};