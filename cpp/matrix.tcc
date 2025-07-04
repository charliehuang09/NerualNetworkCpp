#pragma once

#include "matrix.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <utility>

namespace Matrix {
template <typename T>
Matrix<T>::Matrix()
    : data(nullptr), size(0), shape({.col = 0, .row = 0}),
      stride{.col = 0, .row = 0} {}

template <typename T>
Matrix<T>::Matrix(dimension_t shape)
    : data(new float[shape.col * shape.row]{}), size(shape.col * shape.row),
      shape(shape), stride{.col = shape.row, .row = 1} {}

template <typename T>
Matrix<T>::Matrix(const Matrix &other)
    : data(new float[other.shape.col * other.shape.row]{}),
      size(other.shape.col * other.shape.row), shape(other.shape),
      stride(other.stride) {
  for (int i = 0; i < size; i++) {
    Set(i, other.Get(i));
  }
}

template <typename T>
Matrix<T>::Matrix(Matrix &&other) noexcept
    : data(other.data), size(other.size), shape(other.shape),
      stride(other.stride) {
  other.data = nullptr;
  other.size = 0;
  other.shape = {.col = 0, .row = 0};
  other.stride = {.col = 0, .row = 0};
}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
  if (this != &other) {
    delete[] data;
    data = new float[other.size];
    for (int i = 0; i < other.size; i++) {
      data[i] = other.Get(i);
    }
    size = other.size;
    shape = other.shape;
    stride = other.stride;
  }
  return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) {
  if (this != &other) {
    delete[] data;
    data = other.data;
    size = other.size;
    shape = other.shape;
    stride = other.stride;

    other.data = nullptr;
    other.size = 0;
    other.shape = {.col = 0, .row = 0};
    other.stride = {.col = 0, .row = 0};
  }
  return *this;
}

template <typename T> void Matrix<T>::Print() {
  printf("[");
  for (int i = 0; i < shape.col; i++) {
    printf("[ ");
    for (int j = 0; j < shape.row; j++) {
      printf("%f ", Get(i, j));
    }
    printf("]\n");
  }
  printf("] Height: %d Width: %d\n", shape.col, shape.row);
}

template <typename T> void Matrix<T>::Transpose() {
  std::swap(shape.col, shape.row);
  std::swap(stride.col, stride.row);
}

template <typename T> void Matrix<T>::FillZero() {
  std::memset(data, 0, size * sizeof(float));
}

// Probably want a better way to generate random numbers
template <typename T> void Matrix<T>::FillRand(float min, float max) {
  unsigned int seed;
  std::ifstream rand_source("/dev/random", std::ios::in | std::ios::binary);
  rand_source.read(reinterpret_cast<char *>(&seed), sizeof(seed));

  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dist(min, max);
  for (int i = 0; i < shape.col; i++) {
    for (int j = 0; j < shape.row; j++) {
      Set(i, j, dist(gen));
    }
  }
}

template <typename T> void MatMul(Matrix<T> &a, Matrix<T> &b, Matrix<T> &out) {
  out.FillZero();
  if (a.shape.row != b.shape.col) {
    printf("Assertion failed: a->shape.row != b->shape.col\n");
    printf("Shape of a: (%d, %d)\n", a.shape.row, a.shape.col);
    printf("Shape of b: (%d, %d)\n", b.shape.row, b.shape.col);
    std::abort();
  }
  if (!(out.shape == dimension_t{.col = a.shape.col, .row = b.shape.row})) {
    printf("Outshape is not correct");
    std::abort();
  }
  for (int i = 0; i < a.shape.col; i++) {
    for (int k = 0; k < a.shape.row; k++) {
      for (int j = 0; j < b.shape.row; j++) {
        out.Add(i, j, a.Get(i, k) * b.Get(k, j));
      }
    }
  }
}
template <typename T> void Add(Matrix<T> &a, Matrix<T> &b, Matrix<T> &out) {
  assert(a.shape == b.shape && b.shape == out.shape);
  for (int col = 0; col < a.shape.col; col++) {
    for (int row = 0; row < a.shape.row; row++) {
      out.Set(col, row, a.Get(col, row) + b.Get(col, row));
    }
  }
}

template <typename T>
void Subtract(Matrix<T> &a, Matrix<T> &b, Matrix<T> &out) {
  assert(a.shape == b.shape && b.shape == out.shape);
  for (int col = 0; col < a.shape.col; col++) {
    for (int row = 0; row < a.shape.row; row++) {
      out.Set(col, row, a.Get(col, row) - b.Get(col, row));
    }
  }
}
template <typename T>
void Multiply(Matrix<T> &a, Matrix<T> &b, Matrix<T> &out) {
  assert(a.shape == b.shape && b.shape == out.shape);
  for (int col = 0; col < a.shape.col; col++) {
    for (int row = 0; row < a.shape.row; row++) {
      out.Set(col, row, a.Get(col, row) * b.Get(col, row));
    }
  }
}

template <typename T> void Square(Matrix<T> &a, Matrix<T> &out) {
  assert(a.shape == out.shape);
  for (int col = 0; col < a.shape.col; col++) {
    for (int row = 0; row < a.shape.row; row++) {
      float value = a.Get(col, row);
      out.Set(col, row, value * value);
    }
  }
}
} // namespace Matrix
