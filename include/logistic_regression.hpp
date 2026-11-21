#include "./tensor.hpp"
#include "./linear.hpp"
#include "./optimizer.hpp"

class LogisticRegression
{
    linear *layer;
    Optimizer *optimizer;
    bool binary_classification;

public:
    LogisticRegression(int input_dim, int output_dim = 1, float learning_rate = 0.01, std::string optimizer_type = "SGD");
    ~LogisticRegression();

    Tensor *predict_proba(Tensor *input);

    Tensor *predict(Tensor *input);

    float fit(Tensor *inputs, Tensor *targets, int epochs = 1000, int print_every = 100);

    Tensor *get_weights() { return layer->weights; }
    Tensor *get_bias() { return layer->bias; }
};
