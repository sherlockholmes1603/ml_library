#include "../include/logistic_regression.hpp"

LogisticRegression::LogisticRegression(int input_dim, int output_dim, float learning_rate, std::string optimizer_type)
{
    binary_classification = (output_dim == 1);

    if (optimizer_type != "SGD" && optimizer_type != "Adam")
    {
        throw std::invalid_argument("Unsupported optimizer type: " + optimizer_type);
    }

    layer = new linear(input_dim, output_dim);

    if (optimizer_type == "SGD")
    {
        optimizer = new SGD_NoMomentum(layer->parameters(), learning_rate);
    }
    else // "Adam" — the only remaining possibility after validation
    {
        optimizer = new Adam(layer->parameters(), learning_rate);
    }
}

LogisticRegression::~LogisticRegression()
{
    delete layer;
    delete optimizer;
}

Tensor *LogisticRegression::predict_proba(Tensor *input)
{
    Tensor *logits = layer->forward(input);
    Tensor *probs = binary_classification ? logits->sigmoid() : logits->softmax();

    Tensor *out = new Tensor(probs->data);

    delete probs;
    delete logits;
    return out;
}

Tensor *LogisticRegression::predict(Tensor *input)
{
    Tensor *probabilities = predict_proba(input);
    Tensor *predictions;

    if (binary_classification)
    {
        predictions = new Tensor(probabilities->rows(), 1);
        for (int i = 0; i < probabilities->rows(); ++i)
        {
            (*predictions)(i, 0) = ((*probabilities)(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        }
    }
    else
    {
        predictions = new Tensor(probabilities->rows(), 1);
        for (int i = 0; i < probabilities->rows(); ++i)
        {
            int max_index;
            probabilities->data.row(i).maxCoeff(&max_index);
            (*predictions)(i, 0) = static_cast<float>(max_index);
        }
    }

    delete probabilities;
    return predictions;
}

float LogisticRegression::fit(Tensor *inputs, Tensor *targets, int epochs, int print_every)
{
    float final_loss = 0.0f;
    for (int i = 0; i < epochs; i++)
    {
        Tensor *logits = layer->forward(inputs);                                        // created 1st
        Tensor *y_pred = binary_classification ? logits->sigmoid() : logits->softmax(); // created 2nd
        Tensor *loss = binary_classification
                           ? y_pred->binary_cross_entropy_loss(targets)
                           : y_pred->cross_entropy_loss(targets); // created 3rd

        optimizer->zero_grad();
        loss->backward();
        optimizer->step();

        final_loss = loss->data(0, 0);

        delete loss;
        delete y_pred;
        delete logits;

        if (i % print_every == 0)
        {
            std::cout << "Epoch " << i << ", Loss: " << final_loss << std::endl;
        }
    }

    return final_loss;
}
