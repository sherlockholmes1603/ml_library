#include "linear.hpp"
#include <iostream>
#include <vector>

linear::linear(int input_dim, int output_dim)
{
    weights = new Tensor(input_dim, output_dim, "xavier");
    bias = new Tensor(1, output_dim, "zeros");
}

Tensor *linear::forward(Tensor *input)
{
    Tensor *W = weights;
    Tensor *B = bias;
    Eigen::MatrixXf res = input->data * W->data;
    res.rowwise() += B->data.row(0);

    Tensor *out = new Tensor(res, {input, W, B}, nullptr, "linear");

    out->backward_fn = [input, W, B, out]()
    {
        input->grad += out->grad * W->data.transpose();
        W->grad += input->data.transpose() * out->grad;
        B->grad += out->grad.colwise().sum();
    };

    return out;
}

std::vector<Tensor *> linear::parameters()
{
    return {weights, bias};
}
