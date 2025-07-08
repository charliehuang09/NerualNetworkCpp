#ifndef BIAS_H
#define BIAS_H
#include "layer.h"
namespace Model {

template <typename T> class Bias : public Layer<T> {
private:
  int size_;
  Matrix::Matrix<T> bias_;
  Matrix::Matrix<T> activation_;
  Matrix::Matrix<T> *derrivative_;

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
  Matrix::Matrix<T> *Activation() override { return &activation_; }
  Matrix::Matrix<T> *Derrivative() override { return derrivative_; }
};

} // namespace Model

#include "bias.tcc"

#endif // BIAS_H
