#include "mnist.h"
#include <cstdlib>
#include <sstream>

std::vector<std::string> splitByComma(const std::string &input) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string item = "";

  while (std::getline(ss, item, ',')) {
    result.push_back(item); // Remove whitespace around each item
  }

  return result;
}

MNIST::MNIST() : data_() {}

void MNIST::LoadDataset(std::string path) {
  std::ifstream inputFile(path);
  if (!inputFile.is_open()) {
    std::cerr << "Error: Unable to open file." << std::endl;
    std::abort();
  }

  while (!inputFile.eof()) {
    std::string data;
    std::getline(inputFile, data);

    std::vector<std::string> parsed_data = splitByComma(data);
    if (parsed_data.size() != 785) {
      continue;
    }

    Matrix::Matrix<float> x({.col = 784, .row = 1});
    Matrix::Matrix<float> y({.col = 10, .row = 1}); // TODO
    for (int i = 1; i <= 784; i++) {
      x.Set(i - 1, std::stof(parsed_data[i]) / 255);
    }

    y.FillZero();
    y.Set(std::stoi(parsed_data[0]), 1);

    data_.push_back({.x = x, .y = y});
  }
  inputFile.close();

  std::printf("Number of samples: %lu\n", data_.size());
  return;
}
