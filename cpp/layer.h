#pragma once
#include "matrix.h"
namespace Model {

template <typename T> class Layer {
public:
  virtual ~Layer() = default;
  virtual void InitParam() = 0;
  virtual void Print() = 0;
  virtual void Forward(Matrix::Matrix<T> &input) = 0;
  virtual void Backward(Matrix::Matrix<T> &previous_activation,
                        Matrix::Matrix<T> &next_derrivative) = 0;

  virtual void ApplyDerrivative() = 0;
  virtual void ApplyLearningRate(float lr) = 0;
  virtual Matrix::Matrix<T> *Activation() = 0;
  virtual Matrix::Matrix<T> *Derrivative() = 0;
};
} // namespace Model
