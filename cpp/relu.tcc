#pragma once
#include "relu.h"
#include <cstdio>
#include <cstdlib>
namespace Model {

template <typename T> ReLU<T>::ReLU(int size) : size_(size) {
  derrivative_ = Matrix::Matrix<T>({.col = size, .row = 1});
  activation_ = Matrix::Matrix<T>({.col = size, .row = 1});
}

template <typename T> void ReLU<T>::InitParam() { return; }

template <typename T> void ReLU<T>::InitParam(T min, T max) { return; }

template <typename T> void ReLU<T>::Print() { return; }

template <typename T> void ReLU<T>::Forward(Matrix::Matrix<T> &input) {
  // Apply ReLU function: f(x) = max(0, x)
  for (int i = 0; i < input.size; i++) {
    T value = input.Get(i);
    activation_.Set(i, value > 0 ? value : 0);
  }
}

template <typename T>
void ReLU<T>::Backward(Matrix::Matrix<T> &previous_activation,
                       Matrix::Matrix<T> &next_derrivative) {
  // ReLU derivative: 1 if x > 0, 0 otherwise
  for (int i = 0; i < previous_activation.size; i++) {
    T gradient = previous_activation.Get(i) > 0 ? next_derrivative.Get(i) : 0;
    derrivative_.Set(i, gradient);
  }
}

template <typename T> void ReLU<T>::ApplyDerrivative() {
  // ReLU has no learnable parameters
}

template <typename T> void ReLU<T>::ApplyLearningRate(float lr) {
  // ReLU has no learnable parameters
}
} // namespace Model
