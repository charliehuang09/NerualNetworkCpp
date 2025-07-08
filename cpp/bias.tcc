#pragma once
#include "bias.h"
#include <cstdio>
#include <cstdlib>
namespace Model {

template <typename T>
Bias<T>::Bias(int size) : size_(size), derrivative_(nullptr) {
  bias_ = Matrix::Matrix<T>({.col = size, .row = 1});
  activation_ = Matrix::Matrix<T>({.col = size, .row = 1});
}

template <typename T> void Bias<T>::InitParam() { bias_.FillRand(-1, 1); }

template <typename T> void Bias<T>::InitParam(T min, T max) {
  bias_.FillRand(min, max);
}
template <typename T> void Bias<T>::Print() { bias_.Print(); }

template <typename T> void Bias<T>::Forward(Matrix::Matrix<T> &input) {
  Matrix::Add(input, bias_, activation_);
}

template <typename T>
void Bias<T>::Backward(Matrix::Matrix<T> &previous_activation,
                       Matrix::Matrix<T> &next_derrivative) {
  derrivative_ = &next_derrivative;
}

template <typename T> void Bias<T>::ApplyDerrivative() {
  bias_.Add(*derrivative_);
}

template <typename T> void Bias<T>::ApplyLearningRate(float lr) {
  derrivative_->Multiply(lr);
}
} // namespace Model
