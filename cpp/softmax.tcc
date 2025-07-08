#pragma once
#include "softmax.h"
#include <cstdio>
#include <cstdlib>
namespace Model {

template <typename T> Softmax<T>::Softmax(int size) : size_(size) {
  derrivative_ = Matrix::Matrix<T>({.col = size, .row = 1});
  activation_ = Matrix::Matrix<T>({.col = size, .row = 1});
}

template <typename T> void Softmax<T>::InitParam() { return; }

template <typename T> void Softmax<T>::InitParam(T min, T max) { return; }

template <typename T> void Softmax<T>::Print() { return; }

template <typename T> void Softmax<T>::Forward(Matrix::Matrix<T> &input) {
  float sum = 0.0f;
  for (int i = 0; i < input.size; i++) {
    activation_.Set(i, std::exp(input.Get(i)));
    sum += activation_.Get(i);
  }
  activation_.Divide(sum);
}

template <typename T>
void Softmax<T>::Backward(Matrix::Matrix<T> &previous_activation,
                          Matrix::Matrix<T> &next_derrivative) {
  for (int i = 0; i < derrivative_.size; i++) {
    float grad = 0;
    for (int j = 0; j < derrivative_.size; j++) {
      if (i == j) {
        grad += activation_.Get(i) * (1 - activation_.Get(j)) *
                next_derrivative.Get(j);
      } else {
        grad +=
            -activation_.Get(i) * activation_.Get(j) * next_derrivative.Get(j);
      }
    }
    derrivative_.Set(i, grad);
  }
}

template <typename T> void Softmax<T>::ApplyDerrivative() {}

template <typename T> void Softmax<T>::ApplyLearningRate(float lr) {}
} // namespace Model
