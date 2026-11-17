#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "tensor.hpp"

class linear
{
public:
    Tensor *weights;
    Tensor *bias;

    linear(int input_dim, int output_dim);
    Tensor *forward(Tensor *input);
    std::vector<Tensor *> parameters();

    ~linear()
    {
        delete weights;
        delete bias;
    }
};

#endif // LINEAR_HPP