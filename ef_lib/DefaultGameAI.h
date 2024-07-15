#pragma once

#include <opencv2/opencv.hpp>
#include "GameAI.h"
#include "memory.h"
#include "data.h"



class RetroModelFrameData
{
public:
        RetroModelFrameData(): data(nullptr)
        {
            stack[0] = new cv::Mat;
            stack[1] = new cv::Mat;
            stack[2] = new cv::Mat;
            stack[3] = new cv::Mat;
        }

        ~RetroModelFrameData()
        {
            if(stack[0]) delete stack[0];
            if(stack[1]) delete stack[1];
            if(stack[2]) delete stack[2];
            if(stack[3]) delete stack[3];
        }

        cv::Mat * PushNewFrameOnStack()
        {
            //push everything down
            cv::Mat * tmp = stack[3];
            stack[3] = stack[2];
            stack[2] = stack[1];
            stack[1] = stack[0];
            stack[0] = tmp;

            return stack[0];
        }
        

        void *data;
        unsigned int width;
        unsigned int height;
        unsigned int pitch;
        unsigned int format;

        cv::Mat * stack[4];
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