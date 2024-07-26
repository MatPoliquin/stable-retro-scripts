# TEST adapted from stable-baselines3 docs
# https://stable-baselines3.readthedocs.io/en/master/guide/export.html

import sys
import argparse
import torch as th
import torchvision
from typing import Tuple
import onnx
import onnxruntime as ort
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


#==========================================
# OnnxableSB3Policy
#==========================================
class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)

#==========================================
# parse_cmdline
#==========================================
def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, default='./models/ScoreGoal.zip')
    parser.add_argument('--dest', type=str, default='./models/ScoreGoal')

    print(argv)
    args = parser.parse_args(argv)

    return args

#==========================================
# export_onnx
#==========================================
def export_onnx(args):
    model = PPO.load(args.src, device="cpu")

    onnx_policy = OnnxableSB3Policy(model.policy)

    observation_size = model.observation_space.shape
    print(observation_size)
    dummy_input = th.randn(1, *observation_size)
    dest_onnx_model = args.dest + ".onnx" # "my_ppo_model.onnx"
    
    th.onnx.export(
        onnx_policy,
        dummy_input,
        dest_onnx_model,
        opset_version=17,
        input_names=["input"])
    
    return model, onnx_policy, dummy_input, dest_onnx_model

#==========================================
# export_pytorch
#==========================================
def export_pytorch(args, model, onnx_policy, dummy_input):
    # Pytorch JIT
    # See "ONNX export" for imports and OnnxablePolicy

    # Trace and optimize the module
    traced_module = th.jit.trace(onnx_policy.eval(), dummy_input)
    frozen_module = th.jit.freeze(traced_module)
    frozen_module = th.jit.optimize_for_inference(frozen_module)
    #th.jit.save(frozen_module, jit_path)
    dest_pytorch_model = args.dest + ".pt"
    th.jit.save(traced_module, dest_pytorch_model)

    #test resnet
    #resnet_model = torchvision.models.resnet18()
    #example = th.rand(1, 3, 224, 224)
    #traced_script_module = th.jit.trace(resnet_model, example)
    #traced_script_module.save("traced_resnet_model.pt")

    return dest_pytorch_model

#==========================================
# test_models
#==========================================
def test_models(args, model, dest_onnx_model, dest_pytorch_model):
    ## torch test
    print(th.__version__)
    observation_size = model.observation_space.shape
    dummy_input = th.zeros(1, *observation_size)
    loaded_module = th.jit.load(dest_pytorch_model)
    action_jit = loaded_module(dummy_input)
    print("Pytorch test")
    print(observation_size)
    print(dummy_input)
    print(action_jit)
    print("=========================")

    # onnx test
    onnx_model = onnx.load(dest_onnx_model)
    onnx.checker.check_model(onnx_model)
    observation_size = model.observation_space.shape
    observation = np.zeros((1, *observation_size)).astype(np.float32)
    ort_sess = ort.InferenceSession(dest_onnx_model)
    actions, values, log_prob = ort_sess.run(None, {"input": observation})

    print(actions, values, log_prob)

    # Check that the predictions are the same
    with th.no_grad():
        print(model.policy(th.as_tensor(observation), deterministic=True))


#==========================================
# main
#==========================================
def main(argv):
    
    args = parse_cmdline(argv[1:])

    print("EXPORTING MODELS...")
    model, onnx_policy, dummy_input, dest_onnx_model = export_onnx(args)

    dest_pytorch_model = export_pytorch(args, model, onnx_policy, dummy_input)

    print("DONE EXPORTING MODELS")
    print(dest_onnx_model)
    print(dest_pytorch_model)

    print("TESTING MODELS...")
    test_models(args, model, dest_onnx_model, dest_pytorch_model)


if __name__ == '__main__':
    main(sys.argv)