#ifndef LINEAR_H
#define LINEAR_H
#include "layer.h"
namespace Model {

template <typename T> class Linear : public Layer<T> {
private:
  int in_size_;
  int out_size_;
  Matrix::Matrix<T> weights_;
  Matrix::Matrix<T> activation_;
  Matrix::Matrix<T> derrivative_;
  Matrix::Matrix<T> input_derrivative_;

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
  Matrix::Matrix<T> *Activation() override { return &activation_; }
  Matrix::Matrix<T> *Derrivative() override { return &input_derrivative_; }
};
} // namespace Model

#include "linear.tcc"

#endif // LINEAR_H
