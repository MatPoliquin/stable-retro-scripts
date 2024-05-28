#pragma once

#include "GameAI.h"


class NHL94Data;

class NHL94GameAI : public GameAI {
public:
    virtual void Init(std::filesystem::path dir);

    void SetModelInputs(std::vector<float> & input, const NHL94Data & data);
    virtual void Think(std::bitset<16> & buttons, const Retro::GameData & retro_data);
    void GotoTarget(std::vector<float> & input, int vec_x, int vec_y);

private:
    RetroModel * ScoreGoalModel;
    RetroModel * DefenseModel;
    bool isShooting;
};