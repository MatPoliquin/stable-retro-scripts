#pragma once

#include <opencv2/opencv.hpp>
#include "GameAI.h"
#include "memory.h"
#include "data.h"



class RetroModelFrameData
{
public:
        void *data;
        unsigned int width;
        unsigned int height;
        unsigned int pitch;
        unsigned int format;

        cv::Mat stack[4];
};



class DefaultGameAI : public GameAI {
public:
    virtual void Init(const char * dir, void * ram_ptr, int ram_size);

    virtual void Think(bool buttons[GAMEAI_MAX_BUTTONS], int player, const void *frame_data, unsigned int frame_width, unsigned int frame_height, unsigned int frame_pitch, unsigned int pixel_format);


private:
    RetroModel * model;

    Retro::GameData retro_data;

    RetroModelFrameData input;
};