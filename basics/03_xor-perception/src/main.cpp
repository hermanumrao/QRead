#include "../include/XORnet.h"

#include <cstddef>
#include <iostream>
#include <ostream>
#include <torch/nn/modules/loss.h>
#include <torch/optim/sgd.h>
#include <torch/serialize/input-archive.h>
int main() {
  auto inputs = torch::tensor({{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}});
  auto labels = torch::tensor({{0.0}, {1.0}, {1.0}, {0.0}});

  XORnet net(2);

  net->train();

  auto criterion = torch::nn::MSELoss();
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.1);

  for (size_t epoch = 0; epoch < 100000; ++epoch) {
    optimizer.zero_grad();
    auto output = net->forward(inputs);
    auto loss = criterion(output, labels);
    loss.backward();
    optimizer.step();

    if (epoch % 5000 == 0) {
      std::cout << "Epoch [" << epoch
                << "/10000], Loss: " << loss.item<double>() << std::endl;
    }
  }

  net->eval();

  auto test_output = net->forward(inputs);
  std::cout << "Predicted outputs: " << test_output << std::endl;

  return 0;
}
