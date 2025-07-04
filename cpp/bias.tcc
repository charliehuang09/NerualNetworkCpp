#pragma once
#include "bias.h"
#include <cstdio>
#include <cstdlib>
namespace Model {

template <typename T>
Bias<T>::Bias(int size) : size(size), derrivative(nullptr) {
  bias = Matrix::Matrix<T>({.col = size, .row = 1});
  activation = Matrix::Matrix<T>({.col = size, .row = 1});
}

template <typename T> void Bias<T>::InitParam() { bias.FillRand(-1, 1); }

template <typename T> void Bias<T>::Forward(Matrix::Matrix<T> *input) {
  Matrix::Add(*input, bias, activation);
}

template <typename T>
void Bias<T>::Backward(Matrix::Matrix<T> *previous_activation,
                       Matrix::Matrix<T> *next_derrivative) {
  derrivative = next_derrivative;
}

template <typename T> void Bias<T>::ApplyDerrivative() {
  bias.Add(*derrivative);
}

template <typename T>
// We leave it up to the Layer that owns derrivative to apply the lr
void Bias<T>::ApplyLearningRate(float lr) {
  derrivative->Multiply(lr);
}
} // namespace Model
