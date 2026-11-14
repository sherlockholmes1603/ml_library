#include "linear.hpp"
#include <iostream>
#include <vector>


linear::linear(int input_dim, int output_dim) {
    weights = new Tensor(input_dim, output_dim, "xavier");
    bias = new Tensor(1, output_dim, "zeros");
}

Tensor* linear::forward(Tensor* input) {
    // Dereference the pointers to use the Tensor class overloaded operators
    return *(*input * weights) + bias; 
}

std::vector<Tensor*> linear::parameters() {
    return {weights, bias};
}

