// test of game ai dynamic lib
#include <iostream>
#include <assert.h>
#include <filesystem>
#include "GameAI.h"

#ifdef _WIN32
#include <windows.h> 
#else
#include <dlfcn.h>
#endif


void test_loadlibrary()
{
    GameAI * ga = nullptr;

#ifdef _WIN32
    HINSTANCE hinstLib; 
    BOOL fFreeResult, fRunTimeLinkSuccess = FALSE;

    hinstLib = LoadLibrary(TEXT("game_ai.dll"));
    assert(hinstLib);

    if (hinstLib != NULL) 
    { 
        creategameai_t func  = (creategameai_t) GetProcAddress(hinstLib, "CreateGameAI"); 

        assert(func);
 
        // If the function address is valid, call the function.
        if (NULL != func) 
        {
            fRunTimeLinkSuccess = TRUE;
            ga = func("NHL941on1-Genesis");
            assert(ga);
        }
        // Free the DLL module.
 
        fFreeResult = FreeLibrary(hinstLib); 
    } 
#else
    void *myso = dlopen("./libgame_ai.so", RTLD_NOW);
    assert(myso);
    creategameai_t func = reinterpret_cast<creategameai_t>(dlsym(myso, "CreateGameAI"));
    assert(func);

    ga = func("NHL941on1-Genesis");
    assert(ga);
#endif

    if(ga)
        std::cout << "TEST PASSED!" << std::endl;
}


int main()
{
    test_loadlibrary();

    return 0;
}