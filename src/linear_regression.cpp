#include "linear_regression.hpp"

LinearRegression::LinearRegression(int input_dim, float learning_rate)
{
    layer = new linear(input_dim, 1); // Assuming output_dim is 1 for regression
    std::vector<Tensor *> params = layer->parameters();
    optimizer = new SGD_NoMomentum(params, learning_rate);
}

LinearRegression::~LinearRegression()
{
    delete layer;
    delete optimizer;
}

Tensor *LinearRegression::predict(Tensor *input)
{
    return layer->forward(input);
}

float LinearRegression::fit(Tensor *inputs, Tensor *targets, int epochs, int print_every)
{
    float final_loss = 0.0f;
    for (int i = 0; i < epochs; i++)
    {
        Tensor *y_pred = predict(inputs);
        Tensor *loss = y_pred->mse_loss(targets);

        optimizer->zero_grad();
        loss->backward();
        optimizer->step();

        final_loss = loss->data(0, 0);
        delete y_pred;
        delete loss;

        if (i % print_every == 0)
        {
            std::cout << "Epoch " << i << ", Loss: " << final_loss << std::endl;
        }
    }

    return final_loss;
}
