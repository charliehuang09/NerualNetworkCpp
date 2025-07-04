#pragma once
#include "linear.h"
#include <cstdio>
#include <cstdlib>
namespace Model {

template <typename T>
Linear<T>::Linear(int in_size, int out_size)
    : in_size(in_size), out_size(out_size) {
  weights = Matrix::Matrix<T>({.col = out_size, .row = in_size});
  derrivative = Matrix::Matrix<T>({.col = out_size, .row = in_size});
  activation = Matrix::Matrix<T>({.col = out_size, .row = 1});
  input_derrivative = Matrix::Matrix<T>({.col = in_size, .row = 1});
}

template <typename T> void Linear<T>::InitParam() { weights.FillRand(-1, 1); }

template <typename T> void Linear<T>::Forward(Matrix::Matrix<T> *input) {
  Matrix::MatMul(weights, *input, activation);
}

template <typename T>
void Linear<T>::Backward(Matrix::Matrix<T> *previous_activation,
                         Matrix::Matrix<T> *next_derrivative) {
  previous_activation->Transpose();
  Matrix::MatMul(*next_derrivative, *previous_activation, derrivative);
  previous_activation->Transpose();

  weights.Transpose();
  Matrix::MatMul(weights, *next_derrivative, input_derrivative);
  weights.Transpose();
}

template <typename T> void Linear<T>::ApplyDerrivative() {
  weights.Add(derrivative);
}

template <typename T> void Linear<T>::ApplyLearningRate(float lr) {
  derrivative.Multiply(lr);
}
} // namespace Model
