#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <string>

class Tensor
{
public:
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad; // For storing gradients during backpropagation

    std::vector<Tensor *> parents;
    std::function<void()> backward_fn; // Function to compute gradients during backpropagation
    std::string op;

    // --- Constructors ---
    Tensor(int rows, int cols);
    Tensor(int rows, int cols, std::string init_type); // Initialize with a specific type
    Tensor(Eigen::MatrixXf m,
           std::vector<Tensor *> p = {},
           std::function<void()> bw = nullptr,
           std::string operation = "");

    void backward();

    // --- Accessors ---
    float &operator()(int r, int c);
    float operator()(int r, int c) const;
    int rows() const;
    int cols() const;

    // --- Basic Math (To be implemented in .cpp) ---
    Tensor *operator*(Tensor *other); // Dot product
    Tensor *operator+(Tensor *other);
    Tensor *operator-(Tensor *other);
    Tensor *operator^(Tensor *other); // Hadamard (Element-wise)
    Tensor *operator*(float scalar);

    // --- Transformations & Reductions ---
    Tensor *transpose();
    float sum() const;
    float mean() const;
    // Hint: For Softmax/Loss, you might need sum(axis)
    Tensor *sum(int axis);

    // --- Activations (The "Functions" category) ---
    // These should return a new Tensor with the function applied element-wise
    Tensor *relu();
    Tensor *sigmoid();
    Tensor *tanh();
    Tensor *softmax(); // Usually applied to the last dimension

    float MSE(const Tensor &other) const;                   // Mean Squared Error between this and other
    Tensor *mse_loss(Tensor *target);                       // MSE Loss between this and target, returns a new Tensor representing the loss
    float CrossEntropy(const Tensor &other) const;          // Cross-Entropy Loss between this and other
    Tensor *cross_entropy_loss(Tensor *target);             // Cross-Entropy Loss between this and target, returns a new Tensor representing the loss
    Tensor *softmax_cross_entropy_loss(Tensor *target);     // Combined Softmax + Cross-Entropy for numerical stability
    float binary_cross_entropy(const Tensor &target) const; // Binary Cross-Entropy Loss
    Tensor *binary_cross_entropy_loss(Tensor *target);      // Binary Cross-Entropy Loss, returns a new Tensor representing the loss
    Tensor *mae_loss(Tensor *target);                       // Mean Absolute Error Loss, returns a new Tensor representing the loss

    void zero_grad(); // Utility to reset gradients to zero

    // --- Utilities ---
    void print() const;
};

#endif