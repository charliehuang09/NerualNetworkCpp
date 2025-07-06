#pragma once
#include "layer.h"
namespace Model {

template <typename T> class Bias : public Layer<T> {
private:
  int size;
  Matrix::Matrix<T> bias;
  Matrix::Matrix<T> activation;
  Matrix::Matrix<T> *derrivative;

public:
  Bias(int size);
  ~Bias() = default;
  void InitParam() override;
  void InitParam(T min, T max);
  void Print() override;
  void Forward(Matrix::Matrix<T> &input) override;
  void Backward(Matrix::Matrix<T> &previous_activation,
                Matrix::Matrix<T> &next_derrivative) override;
  void ApplyDerrivative() override;
  void ApplyLearningRate(float lr) override;
  Matrix::Matrix<T> *Activation() override { return &activation; }
  Matrix::Matrix<T> *Derrivative() override { return derrivative; }
};

} // namespace Model

#include "bias.tcc"
