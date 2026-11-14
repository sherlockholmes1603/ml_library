#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "tensor.hpp"
#include <vector>
#include <Eigen/Dense>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    void zero_grad();

protected:
    std::vector<Tensor*> params;
    float lr;
    Optimizer(std::vector<Tensor*> parameters, float learning_rate);
};

class SGD : public Optimizer {
    float momentum;
    std::vector<Eigen::MatrixXf> velocity;
public:
    SGD(std::vector<Tensor*> p, float lr, float m = 0.0f);
    void step() override;
};

class Adam : public Optimizer {
    float beta1, beta2, epsilon;
    int t = 0;
    std::vector<Eigen::MatrixXf> m, v;
public:
    Adam(std::vector<Tensor*> p, float lr, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);
    void step() override;
};

class RMSProp : public Optimizer {
    float alpha, epsilon, momentum;
    std::vector<Eigen::MatrixXf> cache, velocity;
public:
    RMSProp(std::vector<Tensor*> p, float lr, float alpha = 0.99f, float eps = 1e-8f, float momentum = 0.0f);
    void step() override;
};

class Adagrad : public Optimizer {
    float epsilon;
    std::vector<Eigen::MatrixXf> cache;
public:
    Adagrad(std::vector<Tensor*> p, float lr, float eps = 1e-8f);
    void step() override;
};

class SGD_NoMomentum : public Optimizer {
public:
    SGD_NoMomentum(std::vector<Tensor*> p, float lr);
    void step() override;
};

#endif