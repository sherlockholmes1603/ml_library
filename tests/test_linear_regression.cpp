#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/linear_regression.hpp"
#include "../include/tensor.hpp"

class LinearRegressionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
    }
};

TEST_F(LinearRegressionTest, InitializationTest)
{
    int input_dim = 3;
    float lr = 0.01;
    LinearRegression model(input_dim, lr);

    Tensor *weights = model.get_weights();
    Tensor *bias = model.get_bias();

    ASSERT_NE(weights, nullptr);
    ASSERT_NE(bias, nullptr);

    EXPECT_EQ(weights->rows(), input_dim);
    EXPECT_EQ(weights->cols(), 1);
    EXPECT_EQ(bias->rows(), 1);
    EXPECT_EQ(bias->cols(), 1);
}

TEST_F(LinearRegressionTest, TrainingConvergenceTest)
{
    int num_samples = 100;
    int input_dim = 1;

    Tensor *X = new Tensor(num_samples, input_dim);
    X->data.setRandom();

    Tensor *y = new Tensor(num_samples, 1);
    y->data = (X->data * 2.0f).array() + 1.0f;

    LinearRegression model(input_dim, 0.05f);

    float final_loss = model.fit(X, y, 300, 1000);
    EXPECT_LT(final_loss, 0.01f);

    float learned_w = model.get_weights()->data(0, 0);
    float learned_b = model.get_bias()->data(0, 0);
    EXPECT_NEAR(learned_w, 2.0f, 0.1f);
    EXPECT_NEAR(learned_b, 1.0f, 0.1f);

    delete X;
    delete y;
}

TEST_F(LinearRegressionTest, LoopMemoryAccumulationTest)
{
    long long initial_tensors = Tensor::active_tensors;

    // Run your training loop or function
    {
        LinearRegression model(2, 0.01f);
        Tensor X(10, 2);
        Tensor y(10, 1);
        X.data.setRandom();
        y.data.setRandom();

        model.fit(&X, &y, 50, 1000); // 50 epochs
        delete model.get_weights();
        delete model.get_bias();
    }

    // Check that every temporary tensor created during forward/backward
    // passes and epoch loops was successfully destroyed.
    EXPECT_EQ(Tensor::active_tensors, initial_tensors)
        << "Memory leak detected! Tensors left active: " << (Tensor::active_tensors - initial_tensors);
}
