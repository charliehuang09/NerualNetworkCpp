#include "bias.h"
#include "linear.h"
#include "matrix.h"
#include <array>
#include <cstdio>

#define SAMPLES 1
#define EPOCHS 10
#define LOG_FREQ 1
#define LR 0.01
#define M 0.3
#define B 1
#define PRECISION double
int main() {
  std::array<Matrix::Matrix<PRECISION>, SAMPLES> X;
  std::array<Matrix::Matrix<PRECISION>, SAMPLES> Y;
  for (int i = 0; i < SAMPLES; i++) {
    X[i] = Matrix::Matrix<PRECISION>({.col = 1, .row = 1});
    X[i].FillRand(-1, 1);
    Y[i] = Matrix::Matrix(X[i]);
    Y[i].Multiply(M);
    Y[i].Add(B);
  }

  Model::Linear<PRECISION> w1(1, 10);
  Model::Bias<PRECISION> b1(10);
  Model::Linear<PRECISION> w2(10, 10);
  Model::Bias<PRECISION> b2(10);
  Model::Linear<PRECISION> w3(10, 1);
  Model::Bias<PRECISION> b3(1);

  w1.InitParam();
  w2.InitParam();
  w3.InitParam();

  b1.InitParam();
  b2.InitParam();
  b3.InitParam();

  Matrix::Matrix<PRECISION> loss({.col = 1, .row = 1});

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    PRECISION total_loss = 0;
    for (int i = 0; i < SAMPLES; i++) {
      Matrix::Matrix<PRECISION> *x = &X[i];
      Matrix::Matrix<PRECISION> *y = &Y[i];

      w1.Forward(x);
      b1.Forward(w1.Activation());

      w2.Forward(b1.Activation());
      b2.Forward(w2.Activation());

      w3.Forward(b2.Activation());
      b3.Forward(w3.Activation());

      Matrix::Subtract(*y, *b3.Activation(), loss);
      total_loss += loss.Get(0) * loss.Get(0) * 0.5;

      b3.Backward(w3.Activation(), &loss);
      w3.Backward(b2.Activation(), b3.Derrivative());

      b2.Backward(w2.Activation(), w3.Derrivative());
      w2.Backward(w1.Activation(), w3.Derrivative());

      b1.Backward(w1.Activation(), w2.Derrivative());
      w1.Backward(x, w2.Derrivative());

      w3.ApplyLearningRate(LR);
      w2.ApplyLearningRate(LR);
      w1.ApplyLearningRate(LR);

      b3.ApplyLearningRate(LR);
      b2.ApplyLearningRate(LR);
      b1.ApplyLearningRate(LR);

      w3.ApplyDerrivative();
      w2.ApplyDerrivative();
      w1.ApplyDerrivative();

      b3.ApplyDerrivative();
      b2.ApplyDerrivative();
      b1.ApplyDerrivative();
    }
    if (epoch % LOG_FREQ == 0) {
      printf("Epoch: %d Loss: %f\n", epoch, total_loss / SAMPLES);
    }
  }
}

/*
w1 = np.random.rand(2, 1)
b1 = np.random.rand(2, 1)

w2 = np.random.rand(2, 2)
b2 = np.random.rand(2, 1)

w3 = np.random.rand(1, 2)
b3 = np.random.rand(1, 1)

x = X[sample]
y = Y[sample]
a1 = np.matmul(w1, x) + b1
a2 = np.matmul(w2, activation(a1)) + b2
a3 = np.matmul(w3, activation(a2)) + b3

loss = 0.5 * (y - activation(a3)) ** 2

bd3 = (y - activation(a3)) * activation_d(a3)
wd3 = np.matmul(bd3, a2.T)

bd2 = np.matmul(w3.T, bd3) * activation_d(a2)
wd2 = np.matmul(bd2, a1.T)

bd1 = np.matmul(w2.T, bd2) * activation_d(a1)
wd1 = np.matmul(bd1, x.T)

w1 += wd1 * LR
b1 += bd1 * LR

w2 += wd2 * LR
b2 += bd2 * LR

w3 += wd3 * LR
b3 += bd3 * LR
*/
