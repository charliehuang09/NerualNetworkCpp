#include "layer.h"
namespace Model {

class Bias : Model::Layer {
private:
  int size;
  Matrix::Matrix bias;
  Matrix::Matrix activation;
  Matrix::Matrix *derrivative;

public:
  Bias(int size);
  void InitParam();
  void Forward(Matrix::Matrix *input) override;
  void Backward(Matrix::Matrix *previous_activation,
                Matrix::Matrix *next_derrivative) override;
  void ApplyDerrivative() override;
  void ApplyLearningRate(float lr) override;
  Matrix::Matrix *Activation() override { return &activation; }
  Matrix::Matrix *Derrivative() override { return derrivative; }
};

} // namespace Model
