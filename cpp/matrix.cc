#include "matrix.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <random>
#include <utility>

namespace Matrix {

Matrix::Matrix()
    : data(nullptr), size(0), shape({.col = 0, .row = 0}),
      stride{.col = 0, .row = 0} {}

Matrix::Matrix(dimension_t shape)
    : data(new float[shape.col * shape.row]{}), size(shape.col * shape.row),
      shape(shape), stride{.col = shape.row, .row = 1} {}

Matrix::Matrix(const Matrix &other)
    : data(new float[other.shape.col * other.shape.row]{}),
      size(other.shape.col * other.shape.row), shape(other.shape),
      stride(other.stride) {
  for (int i = 0; i < size; i++) {
    Set(i, other.Get(i));
  }
}

Matrix::Matrix(Matrix &&other) noexcept
    : data(other.data), size(other.size), shape(other.shape),
      stride(other.stride) {
  other.data = nullptr;
  other.size = 0;
  other.shape = {.col = 0, .row = 0};
  other.stride = {.col = 0, .row = 0};
}

Matrix &Matrix::operator=(const Matrix &other) {
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

Matrix &Matrix::operator=(Matrix &&other) {
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

void Matrix::Print() {
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

void Matrix::Transpose() {
  std::swap(shape.col, shape.row);
  std::swap(stride.col, stride.row);
}

void Matrix::FillZero() { std::memset(data, 0, size * sizeof(float)); }

// Probably want a better way to generate random numbers
void Matrix::FillRand(float min, float max) {
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

void MatMul(Matrix &a, Matrix &b, Matrix &out) {
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

void Add(Matrix &a, Matrix &b, Matrix &out) {
  assert(a.shape == b.shape && b.shape == out.shape);
  for (int col = 0; col < a.shape.col; col++) {
    for (int row = 0; row < a.shape.row; row++) {
      out.Set(col, row, a.Get(col, row) + b.Get(col, row));
    }
  }
}

void Subtract(Matrix &a, Matrix &b, Matrix &out) {
  assert(a.shape == b.shape && b.shape == out.shape);
  for (int col = 0; col < a.shape.col; col++) {
    for (int row = 0; row < a.shape.row; row++) {
      out.Set(col, row, a.Get(col, row) - b.Get(col, row));
    }
  }
}

void Multiply(Matrix &a, Matrix &b, Matrix &out) {
  assert(a.shape == b.shape && b.shape == out.shape);
  for (int col = 0; col < a.shape.col; col++) {
    for (int row = 0; row < a.shape.row; row++) {
      out.Set(col, row, a.Get(col, row) * b.Get(col, row));
    }
  }
}

void Square(Matrix &a, Matrix &out) {
  assert(a.shape == out.shape);
  for (int col = 0; col < a.shape.col; col++) {
    for (int row = 0; row < a.shape.row; row++) {
      float value = a.Get(col, row);
      out.Set(col, row, value * value);
    }
  }
}

} // namespace Matrix
