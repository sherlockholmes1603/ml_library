#include "../include/logistic_regression.hpp"

LogisticRegression::LogisticRegression(int input_dim, int output_dim, float learning_rate, std::string optimizer_type)
{
    binary_classification = (output_dim == 1);
    layer = new linear(input_dim, output_dim);

    if (optimizer_type == "SGD")
    {
        optimizer = new SGD_NoMomentum(layer->parameters(), learning_rate);
    }
    else if (optimizer_type == "Adam")
    {
        optimizer = new Adam(layer->parameters(), learning_rate);
    }
    else
    {
        throw std::invalid_argument("Unsupported optimizer type: " + optimizer_type);
    }
}

LogisticRegression::~LogisticRegression()
{
    delete layer;
    delete optimizer;
    for (int i = 0; i < cleanup_array.size(); ++i)
    {
        delete cleanup_array[i];
    }
}

Tensor *LogisticRegression::predict_proba(Tensor *input)
{
    Tensor *logits = layer->forward(input);
    this->cleanup_array.push_back(logits);
    if (binary_classification)
    {
        Tensor *probabilities = logits->sigmoid();
        return probabilities;
    }
    else
    {
        // For multi-class classification, apply softmax
        Tensor *probabilities = logits->softmax();
        return probabilities;
    }
}

Tensor *LogisticRegression::predict(Tensor *input)
{
    Tensor *probabilities = predict_proba(input);
    Tensor *predictions;

    if (binary_classification)
    {
        // For binary classification, threshold at 0.5
        predictions = new Tensor(probabilities->rows(), 1);
        for (int i = 0; i < probabilities->rows(); ++i)
        {
            (*predictions)(i, 0) = ((*probabilities)(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        }
    }
    else
    {
        // For multi-class classification, take the argmax
        predictions = new Tensor(probabilities->rows(), 1);
        for (int i = 0; i < probabilities->rows(); ++i)
        {
            int max_index;
            probabilities->data.row(i).maxCoeff(&max_index);
            (*predictions)(i, 0) = static_cast<float>(max_index);
        }
    }

    delete probabilities;
    this->cleanup_array.push_back(predictions);
    return predictions;
}

float LogisticRegression::fit(Tensor *inputs, Tensor *targets, int epochs, int print_every)
{
    float final_loss = 0.0f;
    for (int i = 0; i < epochs; i++)
    {
        Tensor *y_pred = predict_proba(inputs);
        Tensor *loss;

        if (binary_classification)
        {
            loss = y_pred->binary_cross_entropy_loss(targets);
        }
        else
        {
            loss = y_pred->cross_entropy_loss(targets);
        }

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
