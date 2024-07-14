#include "../include/network.h"

#include <iostream>

using namespace torch;

int main(int argc, char *argv[]) {

  Net network(50, 10);
  std::cout << network << '\n' << std::endl;
  Tensor x, output;
  x = torch::randn({50});
  output = network->forward(x);
  std::cout << output;
  return 0;
}
