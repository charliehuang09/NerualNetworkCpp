#include "bias.h"
#include <cstdio>
#include <cstdlib>
namespace Model {

Bias::Bias(int size) : size(size), derrivative(nullptr) {
  bias = Matrix::Matrix({.col = size, .row = 1});
  activation = Matrix::Matrix({.col = size, .row = 1});
}

void Bias::InitParam() { bias.FillRand(-1, 1); }

void Bias::Forward(Matrix::Matrix *input) {
  Matrix::Add(*input, bias, activation);
}

void Bias::Backward(Matrix::Matrix *previous_activation,
                    Matrix::Matrix *next_derrivative) {
  derrivative = next_derrivative;
}

void Bias::ApplyDerrivative() { bias.Add(*derrivative); }

// We leave it up to the Layer that owns derrivative to apply the lr
void Bias::ApplyLearningRate(float lr) { derrivative->Multiply(lr); }
} // namespace Model
