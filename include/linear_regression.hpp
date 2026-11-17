#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "linear.hpp"
#include "optimizer.hpp"

class LinearRegression
{
    linear *layer;
    Optimizer *optimizer;

public:
    LinearRegression(int input_dim, float learning_rate);
    ~LinearRegression();

    Tensor *predict(Tensor *input);
    float fit(Tensor *inputs, Tensor *targets, int epochs = 1000, int print_every = 100);

    Tensor *get_weights() { return layer->weights; }
    Tensor *get_bias() { return layer->bias; }
};

#endif