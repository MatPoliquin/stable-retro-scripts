// test of game ai dynamic lib
#include <iostream>
#include <assert.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "GameAI.h"

#ifdef _WIN32
#include <windows.h> 
#else
#include <dlfcn.h>
#endif

void test_OpenCV()
{
    cv::Mat image;
    cv::Mat grey;
    cv::Mat result;

    image = cv::imread( "../screenshots/wwf.png", cv::IMREAD_COLOR );


    cv::cvtColor(image, grey, cv::COLOR_RGB2GRAY);

    cv::resize(grey, result, cv::Size(84,84), cv::INTER_AREA);
    
    if ( !image.data )
    {
    printf("No image data \n");
    return;
    }
    cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
    cv::imshow("Display Image", result);

    cv::waitKey(0);
}


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

    //std::cout << dlerror() << std::endl;

    assert(myso);
    creategameai_t func = reinterpret_cast<creategameai_t>(dlsym(myso, "CreateGameAI"));
    assert(func);

    if(!func)
        std::cout << "ERROR" << std::endl;

    ga = func("NHL941on1-Genesis");
    assert(ga);

    if(!ga)
        std::cout << "ERROR" << std::endl;
#endif

    std::cout << "HELLO!" << std::endl;

    if(ga)
        std::cout << "TEST PASSED!" << std::endl;
}


int main()
{
    test_loadlibrary();

    test_OpenCV();

    

    return 0;
}