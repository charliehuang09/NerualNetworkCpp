#pragma once
#include "matrix.h"
namespace Model {
class Layer {
public:
  virtual void Forward(Matrix::Matrix *input) = 0;
  virtual void Backward(Matrix::Matrix *previous_activation,
                        Matrix::Matrix *next_derrivative) = 0;

  virtual void ApplyDerrivative() = 0;
  virtual void ApplyLearningRate(float lr) = 0;
  virtual Matrix::Matrix *Activation() = 0;
  virtual Matrix::Matrix *Derrivative() = 0;
};
} // namespace Model
