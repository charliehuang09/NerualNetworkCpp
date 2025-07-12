#include "bias.h"

#include "linear.h"
#include "matrix.h"
#include "mnist.h"
#include "model.h"
#include "relu.h"
#include <chrono>
#include <cstdio>

#define LR 0.01
#define PRECISION float
int main() {

  MNIST dataset;
  dataset.LoadDataset("data/mnist_test.csv");

  Model::Model<PRECISION> model;

  // model.Add(std::make_unique<Model::Linear<PRECISION>>(784, 10));
  // model.Add(std::make_unique<Model::Bias<PRECISION>>(10));
  // model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));
  //
  // model.Add(std::make_unique<Model::Linear<PRECISION>>(10, 10));
  // model.Add(std::make_unique<Model::Bias<PRECISION>>(10));
  // model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));
  //
  // model.Add(std::make_unique<Model::Linear<PRECISION>>(10, 10));
  // model.Add(std::make_unique<Model::Bias<PRECISION>>(10));
  // model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));
  //
  // model.Add(std::make_unique<Model::Linear<PRECISION>>(10, 10));
  // model.Add(std::make_unique<Model::Bias<PRECISION>>(10));
  // model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));
  //
  // model.Add(std::make_unique<Model::Linear<PRECISION>>(10, 10));
  // model.Add(std::make_unique<Model::ReLU<PRECISION>>(10));
  // model.Add(std::make_unique<Model::Bias<PRECISION>>(10));

  auto a = Model::Linear<PRECISION>(784, 10);
  auto b = Model::Bias<PRECISION>(10);
  auto c = Model::ReLU<PRECISION>(10);

  auto d = Model::Linear<PRECISION>(10, 10);
  auto e = Model::Bias<PRECISION>(10);
  auto f = Model::ReLU<PRECISION>(10);

  auto g = Model::Linear<PRECISION>(10, 10);
  auto h = Model::Bias<PRECISION>(10);
  auto i = Model::ReLU<PRECISION>(10);

  auto j = Model::Linear<PRECISION>(10, 10);
  auto k = Model::Bias<PRECISION>(10);
  auto l = Model::ReLU<PRECISION>(10);

  auto m = Model::Linear<PRECISION>(10, 10);
  auto n = Model::Bias<PRECISION>(10);
  auto o = Model::ReLU<PRECISION>(10);

  a.InitParam(-0.3, 0.3);
  b.InitParam(-0.3, 0.3);
  c.InitParam(-0.3, 0.3);
  d.InitParam(-0.3, 0.3);
  e.InitParam(-0.3, 0.3);
  f.InitParam(-0.3, 0.3);
  g.InitParam(-0.3, 0.3);
  h.InitParam(-0.3, 0.3);
  i.InitParam(-0.3, 0.3);
  j.InitParam(-0.3, 0.3);
  k.InitParam(-0.3, 0.3);
  l.InitParam(-0.3, 0.3);
  m.InitParam(-0.3, 0.3);
  n.InitParam(-0.3, 0.3);
  o.InitParam(-0.3, 0.3);

  Matrix::Matrix<PRECISION> loss_derrivative({.col = 10, .row = 1});

  auto start = std::chrono::high_resolution_clock::now();
  for (int idx = 0; idx < dataset.size(); idx++) {
    sample_t data = dataset[idx];
    a.Forward(data.x);
    b.Forward(*a.Activation());
    c.Forward(*b.Activation());
    d.Forward(*c.Activation());
    e.Forward(*d.Activation());
    f.Forward(*e.Activation());
    g.Forward(*f.Activation());
    h.Forward(*g.Activation());
    i.Forward(*h.Activation());
    j.Forward(*i.Activation());
    k.Forward(*j.Activation());
    l.Forward(*k.Activation());
    m.Forward(*l.Activation());
    n.Forward(*m.Activation());
    o.Forward(*n.Activation());

    Matrix::Subtract(data.y, *o.Activation(), loss_derrivative);

    o.Backward(*n.Activation(), loss_derrivative);
    n.Backward(*m.Activation(), *o.Derrivative());
    m.Backward(*m.Activation(), *o.Derrivative());
    l.Backward(*m.Activation(), *o.Derrivative());
    k.Backward(*m.Activation(), *o.Derrivative());
    j.Backward(*m.Activation(), *o.Derrivative());
    i.Backward(*m.Activation(), *o.Derrivative());
    h.Backward(*m.Activation(), *o.Derrivative());
    g.Backward(*m.Activation(), *o.Derrivative());
    f.Backward(*m.Activation(), *o.Derrivative());
    e.Backward(*m.Activation(), *o.Derrivative());
    d.Backward(*m.Activation(), *o.Derrivative());
    c.Backward(*m.Activation(), *o.Derrivative());
    b.Backward(*m.Activation(), *o.Derrivative());
    a.Backward(data.x, *o.Derrivative());

    a.ApplyLearningRate(LR);
    b.ApplyLearningRate(LR);
    c.ApplyLearningRate(LR);
    d.ApplyLearningRate(LR);
    e.ApplyLearningRate(LR);
    f.ApplyLearningRate(LR);
    g.ApplyLearningRate(LR);
    h.ApplyLearningRate(LR);
    i.ApplyLearningRate(LR);
    j.ApplyLearningRate(LR);
    k.ApplyLearningRate(LR);
    l.ApplyLearningRate(LR);
    m.ApplyLearningRate(LR);
    n.ApplyLearningRate(LR);
    o.ApplyLearningRate(LR);

    a.ApplyDerrivative();
    b.ApplyDerrivative();
    c.ApplyDerrivative();
    d.ApplyDerrivative();
    e.ApplyDerrivative();
    f.ApplyDerrivative();
    g.ApplyDerrivative();
    h.ApplyDerrivative();
    i.ApplyDerrivative();
    j.ApplyDerrivative();
    k.ApplyDerrivative();
    l.ApplyDerrivative();
    m.ApplyDerrivative();
    n.ApplyDerrivative();
    o.ApplyDerrivative();
    // model.Backward(loss_derrivative, data.x);
    // model.UpdateParams(LR);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << duration.count() << " seconds\n";
}
