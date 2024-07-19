// test of game ai dynamic lib
#include <iostream>
#include <assert.h>
#include <filesystem>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "GameAI.h"

#ifdef _WIN32
#include <windows.h> 
#else
#include <dlfcn.h>
#endif




std::map<std::string, bool> tests;

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

    std::cout << dlerror() << std::endl;

    assert(myso);

    tests["LOAD LIBRARY"] = true;

    creategameai_t func = reinterpret_cast<creategameai_t>(dlsym(myso, "CreateGameAI"));
    assert(func);

    if(func)
    {
        tests["GET CREATEGAME FUNC"] = true;
        ga = func("./data/NHL941on1-Genesis/NHL941on1.md");
    }

#endif

    if(ga)
        tests["CREATEGAME FUNC"] = true;
}

void Test_Pytorch()
{

    torch::jit::script::Module module;
try {

    module = torch::jit::load("/home/mat/github/stable-retro-scripts/traced_resnet_model.pt");
    //module = torch::jit::load("/home/mat/github/stable-retro-scripts/model.pt");
    std::cerr << "SUCCESS!\n";

    module.eval();

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));
    //inputs.push_back(torch::ones({1, 4, 84, 84}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();

    tests["LOAD PYTORCH MODEL"] = true;
    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  }
  catch (const c10::Error& e) {
    //std::cerr << "error loading the model\n";
    throw std::runtime_error ("error loading the model\n");
    return;
  }
}

int main()
{
    tests.insert(std::pair<std::string, bool>("LOAD LIBRARY",false));
    tests.insert(std::pair<std::string, bool><"GET CREATEGAME FUNC",false>);
    tests.insert(std::pair<std::string, bool><"CREATEGAME FUNC",false>);
    tests.insert(std::pair<std::string, bool><"OPENCV GRAYSCALE DOWNSAMPLE TO 84x84",false>);
    tests.insert(std::pair<std::string, bool><"LOAD PYTORCH MODEL",false>);


    try {
        test_loadlibrary();

        test_OpenCV();

        Test_Pytorch();
    }
    catch (std::exception &e) {
        std::cout << e.what();
    }
    
    return 0;
}