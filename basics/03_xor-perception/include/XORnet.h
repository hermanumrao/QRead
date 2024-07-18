#pragma once

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

struct XORnetImpl : torch::nn::Module {
  XORnetImpl(int fc1_dim) : fc1(fc1_dim, fc1_dim), fc2(fc1_dim, 1) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc1(x));
    x = torch::sigmoid(fc2(x));
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

TORCH_MODULE(XORnet);
