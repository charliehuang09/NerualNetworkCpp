#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "layer.h"
namespace Model {

template <typename T> class Softmax : public Layer<T> {
private:
  int size_;
  Matrix::Matrix<T> activation_;
  Matrix::Matrix<T> derrivative_;

public:
  Softmax(int out_size);
  ~Softmax() = default;
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

#include "softmax.tcc"

#endif // SOFTMAX_H
