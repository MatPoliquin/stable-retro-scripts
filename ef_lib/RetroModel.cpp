#include "RetroModel.h"


//=======================================================
// RetroModelPytorch::LoadModel
//=======================================================
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

//=======================================================
// RetroModelPytorch::Forward
//=======================================================
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


//=======================================================
// RetroModelPytorch::Forward
//=======================================================
void RetroModelPytorch::Forward(std::vector<float> & output, RetroModelFrameData & input)
{
    std::vector<torch::jit::IValue> inputs;

    cv::Mat image(cv::Size(input.width, input.height), CV_8UC2, input.data);
    cv::Mat rgb;  
    cv::Mat gray;
    cv::Mat result;


    cv::cvtColor(image, gray, cv::COLOR_BGR5652GRAY);

    cv::Mat * newFrame = input.PushNewFrameOnStack();

    cv::resize(gray, result, cv::Size(84,84), cv::INTER_AREA);

    result.copyTo(*newFrame);

    //result = result.t();

    /*cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
    cv::imshow("Display Image", result);

    cv::waitKey(0);*/

    at::Tensor test = torch::ones({1, 4, 84, 84});
 
#if 1
    test[0][3] = torch::from_blob(input.stack[0]->data, { result.rows, result.cols }, at::kByte);
    if(input.stack[1]->data)
      test[0][2] = torch::from_blob(input.stack[1]->data, { result.rows, result.cols }, at::kByte);
    if(input.stack[2]->data)
      test[0][1] = torch::from_blob(input.stack[2]->data, { result.rows, result.cols }, at::kByte);
    if(input.stack[3]->data)
      test[0][0] = torch::from_blob(input.stack[3]->data, { result.rows, result.cols }, at::kByte);
#else
    test[0][0] = torch::from_blob(newFrame->data, { result.rows, result.cols }, at::kByte);
    test[0][1] = torch::from_blob(newFrame->data, { result.rows, result.cols }, at::kByte);
    test[0][2] = torch::from_blob(newFrame->data, { result.rows, result.cols }, at::kByte);
    test[0][3] = torch::from_blob(newFrame->data, { result.rows, result.cols }, at::kByte);
#endif

    inputs.push_back(test);

    // Execute the model and turn its output into a tensor.
    torch::jit::IValue ret = module.forward(inputs);
    at::Tensor actions = ret.toTuple()->elements()[0].toTensor();

    for(int i=0; i < 12; i++)
    {
        output[i] = actions[0][i].item<float>();
    }
}

//=======================================================
// TEST
//=======================================================

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
    //module = torch::jit::load("/home/mat/github/stable-retro-scripts/model.pt");
    std::cerr << "SUCCESS!\n";

    module.eval();

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));
    //inputs.push_back(torch::ones({1, 4, 84, 84}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }
}
#endif