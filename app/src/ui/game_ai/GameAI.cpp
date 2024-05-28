#include "GameAI.h"
#include "nhl94.h"
#include <torch/script.h>


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

    //float hello = result[0].size();
    //std::cout << hello << std::endl;

    for(int i=0; i < 12; i++)
    {
        output[i] = result[0][i].item<float>();
    }
}



GameAI * GameAI::CreateGameAI(std::string name)
{
  if(name == "NHL94-Genesis")
  {
    NHL94GameAI * ptr = new NHL94GameAI();
    return ptr;
  }
  
  return NULL;
}

RetroModel * GameAI::LoadModel(std::string path)
{
    RetroModelPytorch * model = new RetroModelPytorch();

    model->LoadModel(path);

    return dynamic_cast<RetroModel*>(model);
}





//=======================================================
// TEST
//======================================================


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

void Test_RetroModel()
{
     torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("/home/mat/github/stable-retro-scripts/ppo_traced.pt");

    //std::cout << module.dump_to_str(true,false, false) << std::endl;

    std::cerr << "SUCCESS!\n";

    //module.eval();

    //auto m = module.named_modules().begin();
    //module.dump(true,true,true);


    //std::cout << TORCH_VERSION_MAJOR << std::endl;

    //std::cout << module.named_children().size() << std::endl;


    std::vector<torch::jit::IValue> inputs;
    std::vector<float> x = {-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0};
    
    //inputs.push_back(torch::zeros({1, 16}));

    at::Tensor test = torch::zeros({1, 16});

    

    test[0][0] = 1.0;
    test[0][1] = 1.0;

    std::cout << test << std::endl;


    inputs.push_back(test);
    

    //std::cout << torch::zeros({1, 16}) << std::endl;

    //std::cout << torch::tensor(x).toString() << std::endl;


    // Execute the model and turn its output into a tensor.
    //auto result = module.forward(inputs);
    c10::IValue result = module.forward(inputs);

    at::Tensor output = result.toTuple()->elements()[0].toTensor();

    std::cout << output << std::endl;
    float o = output[0][0].item<float>();
    std::cout << o << std::endl;

    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/12);

    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/11) << '\n';

  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

}


void GameAI::Test_Pytorch()
{
    //Test_Resnet();

    Test_RetroModel();

}