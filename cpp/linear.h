#pragma once
#include "layer.h"
namespace Model {

template <typename T> class Linear : public Layer<T> {
private:
  int in_size;
  int out_size;
  Matrix::Matrix<T> weights;
  Matrix::Matrix<T> activation;
  Matrix::Matrix<T> derrivative;
  Matrix::Matrix<T> input_derrivative;

public:
  Linear(int in_size, int out_size);
  ~Linear() = default;
  void InitParam() override;
  void InitParam(T min, T max);
  void Print() override;
  void Forward(Matrix::Matrix<T> &input) override;
  void Backward(Matrix::Matrix<T> &previous_activation,
                Matrix::Matrix<T> &next_derrivative) override;
  void ApplyDerrivative() override;
  void ApplyLearningRate(float lr) override;
  Matrix::Matrix<T> *Activation() override { return &activation; }
  Matrix::Matrix<T> *Derrivative() override { return &input_derrivative; }
};
} // namespace Model

#include "linear.tcc"
