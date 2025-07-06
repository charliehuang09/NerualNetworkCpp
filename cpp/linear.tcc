#pragma once
#include "linear.h"
#include <cstdio>
#include <cstdlib>
namespace Model {

template <typename T>
Linear<T>::Linear(int in_size, int out_size)
    : in_size_(in_size), out_size_(out_size) {
  weights_ = Matrix::Matrix<T>({.col = out_size, .row = in_size});
  derrivative_ = Matrix::Matrix<T>({.col = out_size, .row = in_size});
  activation_ = Matrix::Matrix<T>({.col = out_size, .row = 1});
  input_derrivative_ = Matrix::Matrix<T>({.col = in_size, .row = 1});
}

template <typename T> void Linear<T>::InitParam() { weights_.FillRand(-1, 1); }

template <typename T> void Linear<T>::InitParam(T min, T max) {
  weights_.FillRand(min, max);
}

template <typename T> void Linear<T>::Print() { weights_.Print(); }

template <typename T> void Linear<T>::Forward(Matrix::Matrix<T> &input) {
  Matrix::MatMul(weights_, input, activation_);
}

template <typename T>
void Linear<T>::Backward(Matrix::Matrix<T> &previous_activation,
                         Matrix::Matrix<T> &next_derrivative) {
  previous_activation.Transpose();
  Matrix::MatMul(next_derrivative, previous_activation, derrivative_);
  previous_activation.Transpose();

  weights_.Transpose();
  Matrix::MatMul(weights_, next_derrivative, input_derrivative_);
  weights_.Transpose();
}

template <typename T> void Linear<T>::ApplyDerrivative() {
  weights_.Add(derrivative_);
}

template <typename T> void Linear<T>::ApplyLearningRate(float lr) {
  derrivative_.Multiply(lr);
}
} // namespace Model
