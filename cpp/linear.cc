
#include "linear.h"
#include <cstdio>
#include <cstdlib>
namespace Model {
Linear::Linear(int in_size, int out_size)
    : in_size(in_size), out_size(out_size) {
  weights = Matrix::Matrix({.col = out_size, .row = in_size});
  derrivative = Matrix::Matrix({.col = out_size, .row = in_size});
  activation = Matrix::Matrix({.col = out_size, .row = 1});
  input_derrivative = Matrix::Matrix({.col = in_size, .row = 1});
}

void Linear::InitParam() { weights.FillRand(-1, 1); }

void Linear::Forward(Matrix::Matrix *input) {
  Matrix::MatMul(weights, *input, activation);
}

void Linear::Backward(Matrix::Matrix *previous_activation,
                      Matrix::Matrix *next_derrivative) {
  previous_activation->Transpose();
  Matrix::MatMul(*next_derrivative, *previous_activation, derrivative);
  previous_activation->Transpose();

  weights.Transpose();
  Matrix::MatMul(weights, *next_derrivative, input_derrivative);
  weights.Transpose();
}

void Linear::ApplyDerrivative() { weights.Add(derrivative); }

void Linear::ApplyLearningRate(float lr) { derrivative.Multiply(lr); }
} // namespace Model
