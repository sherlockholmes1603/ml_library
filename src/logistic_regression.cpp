#include "../include/logistic_regression.hpp"

LogisticRegression::LogisticRegression(int input_dim, int output_dim, float learning_rate, std::string optimizer_type)
{
    binary_classification = (output_dim == 1);

    // Validate BEFORE allocating anything. If an exception escapes a
    // constructor, C++ does NOT run the destructor (the object was never
    // fully constructed), so everything allocated before the throw leaks.
    // Validate first, allocate after.
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
    // Tier 1 only: the model owns the layer (which owns weights and bias)
    // and the optimizer. Compuation intermediates are NOT the model's job.
    delete layer;
    delete optimizer;
}

Tensor *LogisticRegression::predict_proba(Tensor *input)
{
    // Self-contained: everything created here is deleted here.
    Tensor *logits = layer->forward(input);
    Tensor *probs = binary_classification ? logits->sigmoid() : logits->softmax();

    // Detached copy: no parents -> no dangling pointers, safe for the caller
    // to delete whenever, and safe because inference never calls backward().
    Tensor *out = new Tensor(probs->data);

    delete probs;   // created 2nd -> deleted 1st
    delete logits;  // created 1st -> deleted 2nd
    return out;     // tier 3: caller owns and deletes
}

Tensor *LogisticRegression::predict(Tensor *input)
{
    Tensor *probabilities = predict_proba(input); // detached, self-contained
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
    return predictions; // tier 3: caller owns and deletes
}

float LogisticRegression::fit(Tensor *inputs, Tensor *targets, int epochs, int print_every)
{
    float final_loss = 0.0f;
    for (int i = 0; i < epochs; i++)
    {
        // Build the training graph LOCALLY. Note: fit must NOT call
        // predict_proba for training - its returned tensor is detached
        // (no parents), so backward() would find no path to the weights
        // and training would silently do nothing.
        Tensor *logits = layer->forward(inputs);  // created 1st
        Tensor *y_pred = binary_classification ? logits->sigmoid() : logits->softmax(); // created 2nd
        Tensor *loss = binary_classification
                           ? y_pred->binary_cross_entropy_loss(targets)
                           : y_pred->cross_entropy_loss(targets); // created 3rd

        optimizer->zero_grad();
        loss->backward(); // the whole graph must be alive right here
        optimizer->step();

        final_loss = loss->data(0, 0);

        // Cleanup in REVERSE creation order: children before parents.
        delete loss;    // created 3rd -> deleted 1st
        delete y_pred;  // created 2nd -> deleted 2nd
        delete logits;  // created 1st -> deleted 3rd

        if (i % print_every == 0)
        {
            std::cout << "Epoch " << i << ", Loss: " << final_loss << std::endl;
        }
    }

    return final_loss;
}
