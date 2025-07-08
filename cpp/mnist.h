#include "matrix.h"
#include <utility>

typedef struct Sample {
  Matrix::Matrix<float> x;
  Matrix::Matrix<float> y;
} sample_t;

class MNIST {
public:
  MNIST();
  void LoadDataset(std::string path);
  inline const sample_t &operator[](int index) const { return data_[index]; }
  inline const int size() { return data_.size(); }

private:
  std::vector<sample_t> data_;
};
