#include "DefaultGameAI.h"
#include <cstdlib> 
#include <iostream>
#include <assert.h>
#include <random>

enum DefaultButtons {
    INPUT_B = 0,
    INPUT_A = 1,
    INPUT_MODE = 2,
    INPUT_START = 3,
    INPUT_UP = 4,
    INPUT_DOWN = 5,
    INPUT_LEFT = 6,
    INPUT_RIGHT = 7,
    INPUT_C = 8,
    INPUT_Y = 9,
    INPUT_X = 10,
    INPUT_Z = 11,
    INPUT_MAX = 12
};

void DefaultGameAI::Init(const char * dir, void * ram_ptr, int ram_size)
{
    //std::cout << dir << std::endl;

    std::filesystem::path modelPath = dir;
    modelPath += "/model.pt";
    //modelPath += "/traced_resnet_model.pt";
    std::filesystem::path memDataPath = dir;
    memDataPath += "/data.json";

    model = this->LoadModel(modelPath.string().c_str());

    //retro_data.load()
    std::cout << memDataPath << std::endl;
    retro_data.load(memDataPath.string());
    
    Retro::AddressSpace* m_addressSpace = nullptr;
    m_addressSpace = &retro_data.addressSpace();
	m_addressSpace->reset();
	//Retro::configureData(data, m_core);
	//reconfigureAddressSpace();
    //retro_data.addressSpace().setOverlay(Retro::MemoryOverlay{ '=', '>', 2 });

	m_addressSpace->addBlock(0, ram_size, ram_ptr);
    std::cout << "RAM size:" << ram_size << std::endl;
    std::cout << "RAM ptr:" << ram_ptr << std::endl;

    std::cout << "::Init" << std::endl;
}


void DefaultGameAI::Think(bool buttons[GAMEAI_MAX_BUTTONS], int player, const void *frame_data, unsigned int frame_width, unsigned int frame_height, unsigned int frame_pitch, unsigned int pixel_format)
{

    //std::vector<float> input(16);
    std::vector<float> output(12);

    
    input.data = (void *) frame_data;
    input.width = frame_width;
    input.height = frame_height;
    input.pitch = frame_pitch;
    input.format = pixel_format;

    model->Forward(output, input);

    //std::cout << "::Think" << std::endl;


    for (int i=0; i < output.size(); i++)
    {
        buttons[i] = output[i] >= 1.0 ? 1 : 0;
    }


    buttons[DefaultButtons::INPUT_START] = 0;
    buttons[DefaultButtons::INPUT_MODE] = 0;
    /*buttons[DefaultButtons::INPUT_A] = 0;
    buttons[DefaultButtons::INPUT_B] = 0;
    buttons[DefaultButtons::INPUT_C] = 1;
    buttons[DefaultButtons::INPUT_X] = 0;
    buttons[DefaultButtons::INPUT_Y] = 0;
    buttons[DefaultButtons::INPUT_Z] = 0;

    buttons[DefaultButtons::INPUT_RIGHT] = 1;*/
}