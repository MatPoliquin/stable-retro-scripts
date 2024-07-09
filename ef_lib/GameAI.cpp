#include <torch/script.h>
#include "GameAI.h"
#include "nhl94.h"


#if _WIN32
#define DllExport   __declspec( dllexport )
#else
#define DllExport
#endif


extern "C"  DllExport GameAI * CreateGameAI(const char * name)
{
  //std::cout << name << std::endl;

  if(std::string(name) == "NHL941on1-Genesis")
  {
    NHL94GameAI * ptr = new NHL94GameAI();
    return ptr;
  }
  
  return NULL;
}

extern "C" DllExport int testfunc()
{
  return 8;
}

class RetroModelPytorch : public RetroModel {
public:
        virtual void LoadModel(std::string);
        virtual void Forward(std::vector<float> & output, const std::vector<float> & input);


private:
    torch::jit::script::Module module;
};


void RetroModelPytorch::LoadModel(std::string path)
{
  try {
    this->module = torch::jit::load(path);
    std::cerr << "LOADED MODEL:!" << path << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }
}

void RetroModelPytorch::Forward(std::vector<float> & output, const std::vector<float> & input)
{
    std::vector<torch::jit::IValue> inputs;

    at::Tensor tmp = torch::zeros({1, 16});

    for(int i=0; i < 16; i++)
    {
        tmp[0][i] = input[i];
    }

    inputs.push_back(tmp);
    
    at::Tensor result = module.forward(inputs).toTuple()->elements()[0].toTensor();

    for(int i=0; i < 12; i++)
    {
        output[i] = result[0][i].item<float>();
    }
}


RetroModel * GameAI::LoadModel(const char * path)
{
    RetroModelPytorch * model = new RetroModelPytorch();

    model->LoadModel(std::string(path));

    return dynamic_cast<RetroModel*>(model);
}


void GameAI::DebugPrint(const char * msg)
{
    if (showDebug && debugLogFunc)
    {
        std::cout << msg << std::endl;

        debugLogFunc(0, msg);
    }
}


//=======================================================
// TEST
//======================================================

#if 0
/*
#include "onnxruntime_cxx_api.h"

void Test_ONNX()
{

// Load the model and create InferenceSession
Ort::Env env;
std::string model_path = "path/to/your/onnx/model";
Ort::Session session(env, model_path, Ort::SessionOptions{ nullptr });
// Load and preprocess the input image to inputTensor
...
// Run inference
std::vector outputTensors =
session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 
  inputNames.size(), outputNames.data(), outputNames.size());
const float* outputDataPtr = outputTensors[0].GetTensorMutableData();
std::cout << outputDataPtr[0] << std::endl;


}*/

void Test_Resnet()
{

    torch::jit::script::Module module;
try {

    module = torch::jit::load("/home/mat/github/stable-retro-scripts/traced_resnet_model.pt");
    std::cerr << "SUCCESS!\n";

    module.eval();

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }
}
#endif