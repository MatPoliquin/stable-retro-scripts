#include "nhl94.h"
#include "memory.h"
#include <cstdlib> 
#include <iostream>


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

/*
self.state = (self.game_state.normalized_p1_x, self.game_state.normalized_p1_y, \
                     self.game_state.normalized_p1_velx, self.game_state.normalized_p1_vely, \
                     self.game_state.normalized_p2_x, self.game_state.normalized_p2_y, \
                     self.game_state.normalized_p2_velx, self.game_state.normalized_p2_vely, \
                     self.game_state.normalized_puck_x, self.game_state.normalized_puck_y, \
                     self.game_state.normalized_puck_velx, self.game_state.normalized_puck_vely, \
                     self.game_state.normalized_g2_x, self.game_state.normalized_g2_y, \
                     self.game_state.normalized_player_haspuck, self.game_state.normalized_goalie_haspuck)
*/

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
    DEFENSEZONE_POS_Y = -80
};

/*
        self.normalized_p1_x = self.p1_x / GameConsts.MAX_PLAYER_X
        self.normalized_p1_y = self.p1_y / GameConsts.MAX_PLAYER_Y
        self.normalized_p2_x = self.p2_x / GameConsts.MAX_PLAYER_X
        self.normalized_p2_y = self.p2_y / GameConsts.MAX_PLAYER_Y
        self.normalized_g2_x = self.g2_x / GameConsts.MAX_PLAYER_X
        self.normalized_g2_y = self.g2_y / GameConsts.MAX_PLAYER_Y
        self.normalized_puck_x = self.puck_x / GameConsts.MAX_PUCK_X
        self.normalized_puck_y = self.puck_y / GameConsts.MAX_PUCK_Y
        self.normalized_player_haspuck = 0.0 if self.player_haspuck else 1.0
        self.normalized_goalie_haspuck = 0.0 if self.goalie_haspuck else 1.0


        self.normalized_p1_velx = self.p1_vel_x  / GameConsts.MAX_VEL_XY
        self.normalized_p1_vely = self.p1_vel_y  / GameConsts.MAX_VEL_XY
        self.normalized_p2_velx = self.p2_vel_x  / GameConsts.MAX_VEL_XY
        self.normalized_p2_vely = self.p2_vel_y  / GameConsts.MAX_VEL_XY
        self.normalized_puck_velx = self.puck_vel_x  / GameConsts.MAX_VEL_XY
        self.normalized_puck_vely = self.puck_vel_y  / GameConsts.MAX_VEL_XY

*/

/*
    MAX_PLAYER_X = 120
    MAX_PLAYER_Y = 270

    MAX_PUCK_X = 130
    MAX_PUCK_Y = 270

    MAX_VEL_XY = 50

*/

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


    }
};

void NHL94GameAI::Init(std::filesystem::path dir)
{
    std::cout << dir << std::endl;

    std::filesystem::path scoreModelPath = dir;
    scoreModelPath += "ScoreGoal.pt";
    std::filesystem::path defenseModelPath = dir;
    defenseModelPath += "DefenseZone.pt";


    ScoreGoalModel = this->LoadModel(scoreModelPath);
    DefenseModel = this->LoadModel(defenseModelPath);

    //ScoreGoalModel = this->LoadModel("/home/mat/github/stable-retro-scripts/models/ScoreGoal.pt");
    //DefenseModel = this->LoadModel("/home/mat/github/stable-retro-scripts/models/DefenseZone.pt");

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



void NHL94GameAI::Think(std::bitset<16> & buttons, const Retro::GameData & retro_data)
{
    /*for (auto b=0; b < 16; b++) {
		buttons[b] = std::rand() % 2;
	}*/

    buttons.reset();

    NHL94Data data;
    data.Init(retro_data);

    std::vector<float> input(16);
    std::vector<float> output(12);

    this->SetModelInputs(input, data);

    if (data.p1_haspuck)
    {
        std::cout << "have puck" << std::endl;

        if (data.p1_y >= NHL94Const::ATACKZONE_POS_Y)
        {
            std::cout << "      in attackzone" << std::endl;
            ScoreGoalModel->Forward(output, input);
            output[NHL94Buttons::INPUT_C] = 0;
            output[NHL94Buttons::INPUT_B] = 0;

            if (data.p1_y < 230 && data.p1_y > 210)
            {
                if (data.p1_vel_x >= 30 && data.puck_x > -23 && data.puck_x < 0)
                {
                    std::cout << "Shoot" << std::endl;
                    output[NHL94Buttons::INPUT_C] = 1;
                    isShooting = true;
                }
                else if(data.p1_vel_x <= -30 && data.puck_x < 23 && data.puck_x > 0)
                {
                    std::cout << "Shoot" << std::endl;
                    output[NHL94Buttons::INPUT_C] = 1;
                    isShooting = true;
                }
            }
        }
        else
        {
            this->GotoTarget(output, data.p1_x - 0, data.p1_y - 99);
        }
    }
    else if (data.g1_haspuck)
    {
        output[NHL94Buttons::INPUT_B] = 1;
    }
    else
    {
        std::cout << "Don't have puck" << std::endl;
        isShooting = false;

        if (data.p1_y < NHL94Const::DEFENSEZONE_POS_Y && data.p2_haspuck)
        {
            //std::cout << "    DefenseModel->Forward" << std::endl;
            DefenseModel->Forward(output, input);
        }
        else
        {
            //std::cout << "    GOTO TARGET" << std::endl;            
            GotoTarget(output, data.p1_x - data.puck_x, data.p1_y - data.puck_y);
        }
            
        if (isShooting)
        {
            //output[NHL94Buttons::INPUT_MODE] = 1;
            std::cout << "Shooting" << std::endl;
            output[NHL94Buttons::INPUT_C] = 1;
        }
    }


    
   

    for (int i=0; i < output.size(); i++)
    {
        buttons[i] = output[i] >= 1.0 ? 1 : 0;
    }


    if (buttons[NHL94Buttons::INPUT_B] >= 1 || buttons[NHL94Buttons::INPUT_C] >= 1)
        std::cout << "B,A" << buttons[NHL94Buttons::INPUT_B] << "," << buttons[NHL94Buttons::INPUT_C] << std::endl;


    buttons[NHL94Buttons::INPUT_START] = 0;
    buttons[NHL94Buttons::INPUT_MODE] = 0;
    buttons[NHL94Buttons::INPUT_A] = 0;
    //buttons[NHL94Buttons::INPUT_B] = 0;
    //buttons[NHL94Buttons::INPUT_C] = 0;
    buttons[NHL94Buttons::INPUT_X] = 0;
    buttons[NHL94Buttons::INPUT_Y] = 0;
    buttons[NHL94Buttons::INPUT_Z] = 0;

    

    /*for (const auto& element : input) {
        std::cout << element << " ";
    }

    for (const auto& element : output) {
        std::cout << element << " ";
    }

    std::cout << std::endl;*/

    /*int v_x = data.p2_x - data.p1_x;
    int v_y = data.p2_y - data.p1_y;


    if (v_x > 0)
        buttons[NHL94Buttons::INPUT_RIGHT] = 1;
    else
        buttons[NHL94Buttons::INPUT_LEFT] = 1;

    if (v_y > 0)
        buttons[NHL94Buttons::INPUT_UP] = 1;
    else
        buttons[NHL94Buttons::INPUT_DOWN] = 1;*/



    //std::cout << p2_x.cast<int>() << "," << p2_y.cast<int>() << "\n";


    /*buttons[NHL94Buttons::INPUT_B] = std::rand() % 2;
    buttons[NHL94Buttons::INPUT_A] = std::rand() % 2;
    buttons[NHL94Buttons::INPUT_UP] = std::rand() % 2;
    buttons[NHL94Buttons::INPUT_DOWN] = std::rand() % 2;
    buttons[NHL94Buttons::INPUT_LEFT] = std::rand() % 2;
    buttons[NHL94Buttons::INPUT_RIGHT] = std::rand() % 2;*/

}