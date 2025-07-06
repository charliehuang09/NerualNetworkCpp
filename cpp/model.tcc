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

template <typename T> Model<T>::Model() : m_layers() {}

template <typename T> void Model<T>::Add(std::unique_ptr<Layer<T>> layer) {
  m_layers.push_back(std::move(layer));
}

template <typename T> void Model<T>::InitWeights() {
  for (std::unique_ptr<Layer<T>> &layer : m_layers) {
    layer->InitParam();
  }
}

template <typename T> void Model<T>::Print() {
  for (std::unique_ptr<Layer<T>> &layer : m_layers) {
    layer->Print();
  }
}

template <typename T> void Model<T>::Forward(Matrix::Matrix<T> &x) {
  assert(x.size > 0);
  assert(m_layers.size() > 0);
  m_layers[0]->Forward(x);
  for (size_t i = 1; i < m_layers.size(); i++) {
    m_layers[i]->Forward(*m_layers[i - 1]->Activation());
  }
}

template <typename T> Matrix::Matrix<T> *Model<T>::Output() {
  return m_layers.back().get()->Activation();
}

template <typename T>
void Model<T>::Backward(Matrix::Matrix<T> &loss_derrivative,
                        Matrix::Matrix<T> &x) {
  assert(m_layers.size() != 0);

  if (m_layers.size() == 1) {
    m_layers[m_layers.size() - 1]->Backward(x, loss_derrivative);
    return;
  }

  m_layers[m_layers.size() - 1]->Backward(
      *m_layers[m_layers.size() - 2]->Activation(), loss_derrivative);
  for (int i = m_layers.size() - 2; i >= 1; i--) {
    m_layers[i]->Backward(*m_layers[i - 1]->Activation(),
                          *m_layers[i + 1]->Derrivative());
  }

  m_layers[0]->Backward(x, *m_layers[1]->Derrivative());
}

template <typename T> void Model<T>::UpdateParams(float lr) {
  for (std::unique_ptr<Layer<T>> &layer : m_layers) {
    layer->ApplyLearningRate(lr);
  }
  for (std::unique_ptr<Layer<T>> &layer : m_layers) {
    layer->ApplyDerrivative();
  }
}

template <typename T>
const std::unique_ptr<Layer<T>> &Model<T>::operator[](size_t index) const {
  return m_layers[index];
}

} // namespace Model
