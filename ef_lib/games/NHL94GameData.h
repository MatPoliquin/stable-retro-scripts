#pragma once

#include "memory.h"
#include "../utils/data.h"

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