#include "bias.h"

#include "linear.h"
#include "matrix.h"
#include "mnist.h"
#include "model.h"
#include "relu.h"
#include <chrono>
#include <cstdio>
#include <memory>

#define LR 0.01
#define PRECISION float
int main() {

  MNIST dataset;
  dataset.LoadDataset("data/mnist_test.csv");

  Model::Model<PRECISION> model;

  model.Add(std::make_unique<Model::Linear<PRECISION>>(784, 10));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(10));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(10, 10));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(10));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(10, 10));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(10));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(10, 10));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(10));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(10, 10));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(10));

  model.InitWeights(-0.3, 0.3);

  Matrix::Matrix<PRECISION> loss_derrivative({.col = 10, .row = 1});

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < dataset.size(); i++) {
    sample_t data = dataset[i];
    model.Forward(data.x);

    Matrix::Subtract(data.y, *model.Output(), loss_derrivative);

    model.Backward(loss_derrivative, data.x);
    model.UpdateParams(LR);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << duration.count() << " seconds\n";
}
