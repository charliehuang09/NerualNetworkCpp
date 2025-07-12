#ifndef MATRIX_H
#define MATRIX_H
#include <cassert>
#include <cstdio>

namespace Matrix {
typedef struct Dimension {
  int col;
  int row;
  bool operator==(const Dimension &other) const {
    return col == other.col && row == other.row;
  }
} dimension_t;

// https://numpy.org/doc/stable/dev/internals.html
template <typename T> class Matrix {

private:
public:
  T *data_;
  int size;
  dimension_t shape;
  dimension_t stride; // TODO offset?

  Matrix();
  ~Matrix() { delete[] data_; }
  Matrix(dimension_t shape);
  Matrix(const Matrix &other);
  Matrix(Matrix &&other) noexcept;
  Matrix<T> &operator=(const Matrix<T> &other);
  Matrix<T> &operator=(Matrix<T> &&other);

  void Transpose();
  void Print();
  void FillZero();
  void FillRand(float min, float max);

  inline T Get(int col, int row) const {
    return data_[col * stride.col + row * stride.row];
  }

  inline void Set(int col, int row, T value) {
    data_[col * stride.col + row * stride.row] = value;
  }

  inline T Get(int idx) const { return data_[idx]; }

  inline void Set(int idx, T value) { data_[idx] = value; }

  inline void Add(int col, int row, T value) {
    data_[col * stride.col + row * stride.row] += value;
  }

  inline void Subtract(int col, int row, T value) {
    data_[col * stride.col + row * stride.row] -= value;
  }

  inline void Multiply(int col, int row, T value) {
    data_[col * stride.col + row * stride.row] *= value;
  }

  inline void Divide(int col, int row, T value) {
    data_[col * stride.col + row * stride.row] /= value;
  }

  inline void Add(T value) {
    for (int i = 0; i < size; i++) {
      data_[i] += value;
    }
  }

  inline void Subtract(T value) {
    for (int i = 0; i < size; i++) {
      data_[i] -= value;
    }
  }

  inline void Multiply(T value) {
    for (int i = 0; i < size; i++) {
      data_[i] *= value;
    }
  }

  inline void Divide(T value) {
    for (int i = 0; i < size; i++) {
      data_[i] /= value;
    }
  }

  inline void Add(Matrix &matrix) {
    assert(matrix.shape == shape);
    for (int i = 0; i < shape.col; i++) {
      for (int j = 0; j < shape.row; j++) {
        Add(i, j, matrix.Get(i, j));
      }
    }
  }

  inline void Subtract(Matrix &matrix) {
    assert(matrix.shape == shape);
    for (int i = 0; i < shape.col; i++) {
      for (int j = 0; j < shape.row; j++) {
        Subtract(i, j, matrix.Get(i, j));
      }
    }
  }

  inline void Multiply(Matrix &matrix) {
    assert(matrix.shape == shape);
    for (int i = 0; i < shape.col; i++) {
      for (int j = 0; j < shape.row; j++) {
        Multiply(i, j, matrix.Get(i, j));
      }
    }
  }

  inline void divide(Matrix &matrix) {
    assert(matrix.shape == shape);
    for (int i = 0; i < shape.col; i++) {
      for (int j = 0; j < shape.row; j++) {
        Divide(i, j, matrix.Get(i, j));
      }
    }
  }

  inline void Square() {
    for (int i = 0; i < size; i++) {
      data_[i] *= data_[i];
    }
  }
};
template <typename T> void MatMul(Matrix<T> &a, Matrix<T> &b, Matrix<T> &out);
template <typename T> void Add(Matrix<T> &a, Matrix<T> &b, Matrix<T> &out);
template <typename T> void Subtract(Matrix<T> &a, Matrix<T> &b, Matrix<T> &out);
template <typename T> void Multiply(Matrix<T> &a, Matrix<T> &b, Matrix<T> &out);
template <typename T> void Square(Matrix<T> &a, Matrix<T> &out);

} // namespace Matrix

#include "matrix.tcc"

#endif // MATRIX_H
