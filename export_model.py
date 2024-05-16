# TEST adapted from stable-baselines3 docs
# https://stable-baselines3.readthedocs.io/en/master/guide/export.html

import torch as th
import torchvision
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


MODEL_PATH = "./models/ScoreGoal.zip"

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


# Example: model = PPO("MlpPolicy", "Pendulum-v1")
model = PPO.load(MODEL_PATH, device="cpu")

onnx_policy = OnnxableSB3Policy(model.policy)

observation_size = model.observation_space.shape
print(observation_size)
dummy_input = th.randn(1, *observation_size)
th.onnx.export(
    onnx_policy,
    dummy_input,
    "my_ppo_model.onnx",
    opset_version=17,
    input_names=["input"],
)

# Pytorch JIT
# See "ONNX export" for imports and OnnxablePolicy
jit_path = "ppo_traced.pt"

# Trace and optimize the module
traced_module = th.jit.trace(onnx_policy.eval(), dummy_input)
frozen_module = th.jit.freeze(traced_module)
frozen_module = th.jit.optimize_for_inference(frozen_module)
th.jit.save(frozen_module, jit_path)

#test resnet
resnet_model = torchvision.models.resnet18()
example = th.rand(1, 3, 224, 224)
traced_script_module = th.jit.trace(resnet_model, example)
traced_script_module.save("traced_resnet_model.pt")

##### Load and test with torch
print(th.__version__)

dummy_input = th.zeros(1, *observation_size)
loaded_module = th.jit.load(jit_path)
action_jit = loaded_module(dummy_input)
print("Pytorch test")
print(observation_size)
print(dummy_input)
print(action_jit)
print("=========================")

##### Load and test with onnx

import onnx
import onnxruntime as ort
import numpy as np

onnx_path = "my_ppo_model.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

observation = np.zeros((1, *observation_size)).astype(np.float32)
ort_sess = ort.InferenceSession(onnx_path)
actions, values, log_prob = ort_sess.run(None, {"input": observation})

print(actions, values, log_prob)

# Check that the predictions are the same
with th.no_grad():
    print(model.policy(th.as_tensor(observation), deterministic=True))