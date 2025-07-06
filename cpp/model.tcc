#pragma once
#include "layer.h"
#include "model.h"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <pthread.h>
#include <vector>
namespace Model {

template <typename T> Model<T>::Model() : layers_() {}

template <typename T> void Model<T>::Add(std::unique_ptr<Layer<T>> layer) {
  layers_.push_back(std::move(layer));
}

template <typename T> void Model<T>::InitWeights() {
  for (std::unique_ptr<Layer<T>> &layer : layers_) {
    layer->InitParam();
  }
}

template <typename T> void Model<T>::Print() {
  for (std::unique_ptr<Layer<T>> &layer : layers_) {
    layer->Print();
  }
}

template <typename T> void Model<T>::Forward(Matrix::Matrix<T> &x) {
  assert(x.size > 0);
  assert(layers_.size() > 0);
  layers_[0]->Forward(x);
  for (size_t i = 1; i < layers_.size(); i++) {
    layers_[i]->Forward(*layers_[i - 1]->Activation());
  }
}

template <typename T> Matrix::Matrix<T> *Model<T>::Output() {
  return layers_.back().get()->Activation();
}

template <typename T>
void Model<T>::Backward(Matrix::Matrix<T> &loss_derrivative,
                        Matrix::Matrix<T> &x) {
  assert(layers_.size() != 0);

  if (layers_.size() == 1) {
    layers_[layers_.size() - 1]->Backward(x, loss_derrivative);
    return;
  }

  layers_[layers_.size() - 1]->Backward(
      *layers_[layers_.size() - 2]->Activation(), loss_derrivative);
  for (int i = layers_.size() - 2; i >= 1; i--) {
    layers_[i]->Backward(*layers_[i - 1]->Activation(),
                         *layers_[i + 1]->Derrivative());
  }

  layers_[0]->Backward(x, *layers_[1]->Derrivative());
}

template <typename T> void Model<T>::UpdateParams(float lr) {
  for (std::unique_ptr<Layer<T>> &layer : layers_) {
    layer->ApplyLearningRate(lr);
  }
  for (std::unique_ptr<Layer<T>> &layer : layers_) {
    layer->ApplyDerrivative();
  }
}

template <typename T>
const std::unique_ptr<Layer<T>> &Model<T>::operator[](size_t index) const {
  return layers_[index];
}

} // namespace Model
