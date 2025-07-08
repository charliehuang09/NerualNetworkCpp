#ifndef RELU_H
#define RELU_H
#include "layer.h"
namespace Model {

template <typename T> class ReLU : public Layer<T> {
private:
  int size_;
  Matrix::Matrix<T> activation_;
  Matrix::Matrix<T> derrivative_;

public:
  ReLU(int out_size);
  ~ReLU() = default;
  void InitParam() override;
  void InitParam(T min, T max) override;
  void Print() override;
  void Forward(Matrix::Matrix<T> &input) override;
  void Backward(Matrix::Matrix<T> &previous_activation,
                Matrix::Matrix<T> &next_derrivative) override;
  void ApplyDerrivative() override;
  void ApplyLearningRate(float lr) override;
  Matrix::Matrix<T> *Activation() override { return &activation_; }
  Matrix::Matrix<T> *Derrivative() override { return &derrivative_; }
};
} // namespace Model

#include "relu.tcc"

#endif // RELU_H
