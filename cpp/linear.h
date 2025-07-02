#include "layer.h"
namespace Model {
class Linear : Layer {
private:
  int in_size;
  int out_size;
  Matrix::Matrix weights;
  Matrix::Matrix activation;
  Matrix::Matrix derrivative;
  Matrix::Matrix input_derrivative;

public:
  Linear(int in_size, int out_size);
  void InitParam();
  void Forward(Matrix::Matrix *input) override;
  void Backward(Matrix::Matrix *previous_activation,
                Matrix::Matrix *next_derrivative) override;
  void ApplyDerrivative() override;
  void ApplyLearningRate(float lr) override;
  Matrix::Matrix *Activation() override { return &activation; }
  Matrix::Matrix *Derrivative() override { return &input_derrivative; }
};
} // namespace Model
