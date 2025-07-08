#ifndef MODEL_H
#define MODEL_H
#include "layer.h"
#include <memory>
#include <vector>
namespace Model {

template <typename T> class Model {

public:
  Model();
  void Add(std::unique_ptr<Layer<T>> layer);
  void InitWeights();
  void InitWeights(T min, T max);
  void Print();
  void Forward(Matrix::Matrix<T> &x);
  Matrix::Matrix<T> *Output();
  void Backward(Matrix ::Matrix<T> &loss_derrivative, Matrix ::Matrix<T> &x);
  void UpdateParams(float lr);

  const std::unique_ptr<Layer<T>> &operator[](size_t index) const;

private:
  std::vector<std::unique_ptr<Layer<T>>> layers_;
};
} // namespace Model

#include "model.tcc"

#endif // MODEL_H
