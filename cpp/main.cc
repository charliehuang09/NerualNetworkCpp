#include "bias.h"
#include "linear.h"
#include "matrix.h"
#include "mnist.h"
#include "model.h"
#include "relu.h"
#include "softmax.h"
#include <cstdio>
#include <memory>

#define EPOCHS 100
#define LOG_FREQ 1
#define LR 0.01
#define PRECISION float
int main() {

  MNIST train, test;
  train.LoadDataset("data/mnist_train.csv");
  test.LoadDataset("data/mnist_test.csv");

  Model::Model<PRECISION> model;

  model.Add(std::make_unique<Model::Linear<PRECISION>>(784, 64));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(64));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(64));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(64, 64));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(64));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(64));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(64, 64));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(64));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(64));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(64, 64));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(64));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(64));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(64, 64));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(64));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(64));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(64, 64));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(64));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(64));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(64, 64));
  model.Add(std::make_unique<Model::Bias<PRECISION>>(64));
  model.Add(std::make_unique<Model::ReLU<PRECISION>>(64));

  model.Add(std::make_unique<Model::Linear<PRECISION>>(64, 10));
  model.Add(std::make_unique<Model::Softmax<PRECISION>>(10));

  model.InitWeights(-0.2, 0.2);

  Matrix::Matrix<PRECISION> loss_derrivative({.col = 10, .row = 1});

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    PRECISION total_loss = 0;
    for (int i = 0; i < train.size(); i++) {
      sample_t data = train[i];
      model.Forward(data.x);

      Matrix::Subtract(data.y, *model.Output(), loss_derrivative);
      for (int i = 0; i < model.Output()->size; i++) {
        if (data.y.Get(i) == 1) {
          PRECISION pred = model.Output()->Get(i);
          total_loss += -std::log(std::max(pred, 1e-10f));
          break;
        }
      }

      model.Backward(loss_derrivative, data.x);
      model.UpdateParams(LR);
    }

    if (epoch % LOG_FREQ == 0) {
      printf("Epoch: %d Loss: %f\n", epoch, total_loss / train.size());
    }
  }

  PRECISION num_right = 0;
  for (int i = 0; i < test.size(); i++) {
    sample_t data = test[i];
    model.Forward(data.x);

    int max_idx = 0;
    PRECISION max = -1;
    for (int i = 0; i < data.y.size; i++) {
      if (model.Output()->Get(i) > max) {
        max_idx = i;
        max = model.Output()->Get(i);
      }
    }
    if (data.y.Get(max_idx) == 1) {
      num_right++;
    }
    model.Backward(loss_derrivative, data.x);
    model.UpdateParams(LR);
  }
  printf("Number correct: %f", num_right / train.size());
}
