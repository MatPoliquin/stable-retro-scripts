// test of game ai dynamic lib
#include <iostream>
#include <assert.h>
#include <dlfcn.h>
#include <filesystem>
#include "GameAI.h"


void test_loadlibrary()
{
    void *myso = dlopen("./libgame_ai.so", RTLD_NOW);
    assert(myso);
    creategameai_t func = reinterpret_cast<creategameai_t>(dlsym(myso, "CreateGameAI"));
    assert(func);

    GameAI * ga = func("NHL941on1-Genesis");
}


int main()
{
    std::cout << "HELLO!!" << std::endl;

    test_loadlibrary();

    return 0;
}