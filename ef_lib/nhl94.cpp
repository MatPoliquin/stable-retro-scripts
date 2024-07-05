#include "nhl94.h"
#include <cstdlib> 
#include <iostream>
#include <assert.h>

enum NHL94Buttons {
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

enum NHL94NeuralNetInput {
    P1_X = 0,
    P1_Y,
    P1_VEL_X,
    P1_VEL_Y,
    P2_X,
    P2_Y,
    P2_VEL_X,
    P2_VEL_Y,
    PUCK_X,
    PUCK_Y,
    PUCK_VEL_X,
    PUCK_VEL_Y,
    G2_X,
    G2_Y,
    P1_HASPUCK,
    G1_HASPUCK,
    NN_INPUT_MAX,

    // Used for normalization
    MAX_PLAYER_X = 120,
    MAX_PLAYER_Y = 270,
    MAX_PUCK_X = 130,
    MAX_PUCK_Y = 270,
    MAX_VEL_XY = 50
};


enum NHL94Const {
    ATACKZONE_POS_Y = 100,
    DEFENSEZONE_POS_Y = -80,
    SCORE_ZONE_TOP = 230,
    SCORE_ZONE_BOTTOM = 210,
};

class NHL94Data {
public:
    int p1_x;
    int p1_y;
    int p2_x;
    int p2_y;
    int p1_vel_x;
    int p1_vel_y;
    int p2_vel_x;
    int p2_vel_y;
    int g1_x;
    int g1_y;
    int g2_x;
    int g2_y;

    int puck_x;
    int puck_y;
    int puck_vel_x;
    int puck_vel_y;

    int p1_fullstar_x;
    int p1_fullstar_y;
    int p2_fullstar_x;
    int p2_fullstar_y;


    bool p1_haspuck;
    bool g1_haspuck;
    bool p2_haspuck;
    bool g2_haspuck;

    int attack_zone_y;
    int defense_zone_y;
    int score_zone_top;
    int score_zone_bottom;

    int period;

    void Init(const Retro::GameData & data)
    {
        // players
        p1_x = data.lookupValue("p1_x").cast<int>();
        p1_y = data.lookupValue("p1_y").cast<int>();
        p2_x = data.lookupValue("p2_x").cast<int>();
        p2_y = data.lookupValue("p2_y").cast<int>();
        p1_vel_x = data.lookupValue("p1_vel_x").cast<int>();
        p1_vel_y = data.lookupValue("p1_vel_y").cast<int>();
        p2_vel_x = data.lookupValue("p2_vel_x").cast<int>();
        p2_vel_y = data.lookupValue("p2_vel_y").cast<int>();

        // goalies
        g1_x = data.lookupValue("g1_x").cast<int>();
        g1_y = data.lookupValue("g1_y").cast<int>();
        g2_x = data.lookupValue("g2_x").cast<int>();
        g2_y = data.lookupValue("g2_y").cast<int>();

        // puck
        puck_x = data.lookupValue("puck_x").cast<int>();
        puck_y = data.lookupValue("puck_y").cast<int>();
        puck_vel_x = data.lookupValue("puck_vel_x").cast<int>();
        puck_vel_y = data.lookupValue("puck_vel_y").cast<int>();

        p1_fullstar_x = data.lookupValue("fullstar_x").cast<int>();
        p1_fullstar_y = data.lookupValue("fullstar_y").cast<int>();
        p2_fullstar_x = data.lookupValue("p2_fullstar_x").cast<int>();
        p2_fullstar_y = data.lookupValue("p2_fullstar_y").cast<int>();

        period = data.lookupValue("period").cast<int>();


        // Knowing if the player has the puck is tricky since the fullstar in the game is not aligned with the player every frame
        // There is an offset of up to 2 sometimes

        if (std::abs(p1_x - p1_fullstar_x) < 3 && std::abs(p1_y - p1_fullstar_y) < 3)
            p1_haspuck = true;
        else
            p1_haspuck = false;

        if(std::abs(p2_x - p1_fullstar_x) < 3 && std::abs(p2_y - p1_fullstar_y) < 3)
            p2_haspuck = true;
        else
            p2_haspuck = false;
            
        if(std::abs(g1_x - p1_fullstar_x) < 3 && std::abs(g1_y - p1_fullstar_y) < 3)
            g1_haspuck = true;
        else
            g1_haspuck = false;

        if(std::abs(g2_x - p1_fullstar_x) < 3 && std::abs(g2_y - p1_fullstar_y) < 3)
            g2_haspuck = true;
        else
            g2_haspuck = false;


        attack_zone_y = NHL94Const::ATACKZONE_POS_Y;
        defense_zone_y = NHL94Const::DEFENSEZONE_POS_Y;
        score_zone_top = NHL94Const::SCORE_ZONE_TOP;
        score_zone_bottom = NHL94Const::SCORE_ZONE_BOTTOM;

        //std::cout << p1_x << "," << p1_y << "/" << puck_x << "," << puck_y << std::endl;
    }

    void Flip()
    {
        std::swap(p1_x, p2_x);
        std::swap(p1_y, p2_y);
        std::swap(g1_x, g2_x);
        std::swap(g1_y, g2_y);
        std::swap(p1_haspuck, p2_haspuck);
        std::swap(g1_haspuck, g2_haspuck);

        std::swap(p1_vel_x, p2_vel_x);
        std::swap(p1_vel_y, p2_vel_y);
    }

    void FlipZones()
    {
        p1_x = -p1_x;
        p1_y = -p1_y;
        p2_x = -p2_x;
        p2_y = -p2_y;
        g1_x = -g1_x;
        g1_y = -g1_y;
        g2_x = -g2_x;
        g2_y = -g2_y;

        p1_vel_x = -p1_vel_x;
        p1_vel_y = -p1_vel_y;
        p2_vel_x = -p2_vel_x;
        p2_vel_y = -p2_vel_y;

        puck_x = -puck_x;
        puck_y = -puck_y;

        puck_vel_x = -puck_vel_x;
        puck_vel_y = -puck_vel_y;


        /*
        attack_zone_y = -attack_zone_y;
        defense_zone_y = -defense_zone_y;

        std::swap(score_zone_top, score_zone_bottom);
        score_zone_top = -score_zone_top;
        score_zone_bottom = -score_zone_bottom;
        */
    }
};

void NHL94GameAI::Init(const char * dir, void * ram_ptr, int ram_size)
{
    //std::cout << dir << std::endl;

    std::filesystem::path scoreModelPath = dir;
    scoreModelPath += "/ScoreGoal.pt";
    std::filesystem::path defenseModelPath = dir;
    defenseModelPath += "/DefenseZone.pt";
    std::filesystem::path memDataPath = dir;
    memDataPath += "/data.json";
    std::filesystem::path sysDataPath = dir;
    sysDataPath += "/sys.json";

    ScoreGoalModel = this->LoadModel(scoreModelPath.string().c_str());
    DefenseModel = this->LoadModel(defenseModelPath.string().c_str());

    //retro_data.load()
    std::cout << memDataPath << std::endl;
    retro_data.load(memDataPath.string());
    
    Retro::AddressSpace* m_addressSpace = nullptr;
    m_addressSpace = &retro_data.addressSpace();
	m_addressSpace->reset();
	//Retro::configureData(data, m_core);
	//reconfigureAddressSpace();
    retro_data.addressSpace().setOverlay(Retro::MemoryOverlay{ '=', '>', 2 });

	m_addressSpace->addBlock(16711680, ram_size, ram_ptr);
    std::cout << "RAM size:" << ram_size << std::endl;
    std::cout << "RAM ptr:" << ram_ptr << std::endl;
    
    static_assert(NHL94NeuralNetInput::NN_INPUT_MAX == 16);

    isShooting = false;
}

void NHL94GameAI::SetModelInputs(std::vector<float> & input, const NHL94Data & data)
{
    // players
    input[NHL94NeuralNetInput::P1_X] = (float)data.p1_x / (float) NHL94NeuralNetInput::MAX_PLAYER_X;
    input[NHL94NeuralNetInput::P1_Y] = (float)data.p1_y / (float) NHL94NeuralNetInput::MAX_PLAYER_Y;
    input[NHL94NeuralNetInput::P2_X] = (float)data.p2_x / (float) NHL94NeuralNetInput::MAX_PLAYER_X;
    input[NHL94NeuralNetInput::P2_Y] = (float) data.p2_y / (float) NHL94NeuralNetInput::MAX_PLAYER_Y;
    input[NHL94NeuralNetInput::G2_X] = (float) data.g2_x / (float) NHL94NeuralNetInput::MAX_PLAYER_X;
    input[NHL94NeuralNetInput::G2_Y] = (float) data.g2_y / (float) NHL94NeuralNetInput::MAX_PLAYER_Y;
    input[NHL94NeuralNetInput::P1_VEL_X] = (float) data.p1_vel_x / (float) NHL94NeuralNetInput::MAX_VEL_XY;
    input[NHL94NeuralNetInput::P1_VEL_Y] = (float) data.p1_vel_y / (float) NHL94NeuralNetInput::MAX_VEL_XY;
    input[NHL94NeuralNetInput::P2_VEL_X] = (float) data.p2_vel_x / (float) NHL94NeuralNetInput::MAX_VEL_XY;
    input[NHL94NeuralNetInput::P2_VEL_Y] = (float) data.p2_vel_y / (float) NHL94NeuralNetInput::MAX_VEL_XY;

    // puck
    input[NHL94NeuralNetInput::PUCK_X] = (float) data.puck_x / (float) NHL94NeuralNetInput::MAX_PLAYER_X;
    input[NHL94NeuralNetInput::PUCK_Y] = (float) data.puck_y / (float) NHL94NeuralNetInput::MAX_PLAYER_Y;
    input[NHL94NeuralNetInput::PUCK_VEL_X] = (float) data.puck_vel_x / (float) NHL94NeuralNetInput::MAX_VEL_XY;
    input[NHL94NeuralNetInput::PUCK_VEL_Y] = (float) data.puck_vel_y / (float) NHL94NeuralNetInput::MAX_VEL_XY;

    input[NHL94NeuralNetInput::P1_HASPUCK] = data.p1_haspuck ? 0.0 : 1.0;
    input[NHL94NeuralNetInput::G1_HASPUCK] = data.g1_haspuck ? 0.0 : 1.0; 
}

void NHL94GameAI::GotoTarget(std::vector<float> & input, int vec_x, int vec_y)
{
    if (vec_x > 0)
        input[NHL94Buttons::INPUT_LEFT] = 1;
    else
        input[NHL94Buttons::INPUT_RIGHT] = 1;

    if (vec_y > 0)
        input[NHL94Buttons::INPUT_DOWN] = 1;
    else
        input[NHL94Buttons::INPUT_UP] = 1;
}

bool isInsideAttackZone(NHL94Data & data)
{
    if (data.attack_zone_y > 0 && data.p1_y >= data.attack_zone_y)
    {
        return true;
    }
    else if (data.attack_zone_y < 0 && data.p1_y <= data.attack_zone_y)
    {
        return true;
    }

    return false;
}

bool isInsideScoreZone(NHL94Data & data)
{
    if (data.p1_y < data.score_zone_top && data.p1_y > data.score_zone_bottom)
    {
        return true;
    }
    
    return false;
}

bool isInsideDefenseZone(NHL94Data & data)
{
    if (data.defense_zone_y > 0 && data.p1_y >= data.defense_zone_y)
    {
        return true;
    }
    else if (data.defense_zone_y < 0 && data.p1_y <= data.defense_zone_y)
    {
        return true;
    }

    return false;
}

void DebugPrint(const char * msg)
{
    std::cout << msg << std::endl;
}

void NHL94GameAI::Think(bool buttons[GAMEAI_MAX_BUTTONS], int player)
{
    NHL94Data data;
    data.Init(retro_data);

    data.Flip();

    if(data.period % 2 == 0)
    {
        data.FlipZones();
    }

    std::vector<float> input(16);
    std::vector<float> output(12);

    this->SetModelInputs(input, data);

    if (data.p1_haspuck)
    {
        DebugPrint("have puck");

        if (isInsideAttackZone(data))
        {
            DebugPrint("      in attackzone");
            ScoreGoalModel->Forward(output, input);
            output[NHL94Buttons::INPUT_C] = 0;
            output[NHL94Buttons::INPUT_B] = 0;

            if (isInsideScoreZone(data))
            {
                if (data.p1_vel_x >= 30 && data.puck_x > -23 && data.puck_x < 0)
                {
                    DebugPrint("Shoot");
                    output[NHL94Buttons::INPUT_C] = 1;
                    isShooting = true;
                }
                else if(data.p1_vel_x <= -30 && data.puck_x < 23 && data.puck_x > 0)
                {
                    DebugPrint("Shoot");
                    output[NHL94Buttons::INPUT_C] = 1;
                    isShooting = true;
                }
            }
        }
        else
        {
            this->GotoTarget(output, data.p1_x, -data.attack_zone_y);
        }
    }
    else if (data.g1_haspuck)
    {
        output[NHL94Buttons::INPUT_B] = 1;
    }
    else
    {
        DebugPrint("Don't have puck");
        isShooting = false;

        if (isInsideDefenseZone(data) && data.p2_haspuck)
        {
            DebugPrint("    DefenseModel->Forward");
            DefenseModel->Forward(output, input);
        }
        else
        {
            DebugPrint("    GOTO TARGET");            
            GotoTarget(output, data.p1_x - data.puck_x, data.p1_y - data.puck_y);
        }
            
        if (isShooting)
        {
            //output[NHL94Buttons::INPUT_MODE] = 1;
            DebugPrint("Shooting");
            output[NHL94Buttons::INPUT_C] = 1;
        }
    }

    assert(output.size() <= 16);
    for (int i=0; i < output.size(); i++)
    {
        buttons[i] = output[i] >= 1.0 ? 1 : 0;
    }

   
    buttons[NHL94Buttons::INPUT_START] = 0;
    buttons[NHL94Buttons::INPUT_MODE] = 0;
    buttons[NHL94Buttons::INPUT_A] = 0;
    //buttons[NHL94Buttons::INPUT_B] = 0;
    //buttons[NHL94Buttons::INPUT_C] = 0;
    buttons[NHL94Buttons::INPUT_X] = 0;
    buttons[NHL94Buttons::INPUT_Y] = 0;
    buttons[NHL94Buttons::INPUT_Z] = 0;

    //Flip directions
    if(data.period % 2 == 0)
    {
        std::swap(buttons[NHL94Buttons::INPUT_UP], buttons[NHL94Buttons::INPUT_DOWN]);
        std::swap(buttons[NHL94Buttons::INPUT_LEFT], buttons[NHL94Buttons::INPUT_RIGHT]);
    }
}