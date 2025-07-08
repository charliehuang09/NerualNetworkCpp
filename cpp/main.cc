#include "bias.h"
#include "linear.h"
#include "matrix.h"
#include "mnist.h"
#include "model.h"
#include "relu.h"
#include <cstdio>
#include <memory>

#define SAMPLES 10
#define EPOCHS 1000
#define LOG_FREQ 10
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

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    PRECISION total_loss = 0;
    for (int i = 0; i < dataset.size(); i++) {
      sample_t data = dataset[i];
      model.Forward(data.x);

      Matrix::Subtract(data.y, *model.Output(), loss_derrivative);
      for (int i = 0; i < loss_derrivative.size; i++) {
        total_loss +=
            (loss_derrivative.Get(i) * loss_derrivative.Get(i) * 0.5) / 10;
      }

      model.Backward(loss_derrivative, data.x);
      model.UpdateParams(LR);
    }
    if (epoch % LOG_FREQ == 0) {
      printf("Epoch: %d Loss: %f\n", epoch, total_loss / SAMPLES);
    }
  }
}
