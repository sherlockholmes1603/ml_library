#include <gtest/gtest.h>
#include "../include/optimizer.hpp"

class OptimizerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(OptimizerTest, ZeroGradWorks)
{
    Tensor *t = new Tensor(2, 2);
    t->data << 1.0f, 2.0f, 3.0f, 4.0f;
    t->grad << 0.5f, 0.5f, 0.5f, 0.5f;
    t->zero_grad();

    std::vector<Tensor *> params = {t};
    SGD_NoMomentum Optimizer(params, 0.1f);

    Optimizer.zero_grad();

    EXPECT_TRUE(t->grad.isApprox(Eigen::MatrixXf::Zero(2, 2)));

    delete t;
}

TEST_F(OptimizerTest, SGDNoMomentumStep)
{
    Tensor *t = new Tensor(1, 2);
    t->data << 10.0f, 20.0f;
    t->grad << 1.0f, 2.0f;

    std::vector<Tensor *> params = {t};
    float lr = 0.1f;
    SGD_NoMomentum Optimizer(params, lr);

    Optimizer.step();
    Eigen::MatrixXf expected_data(1, 2);
    expected_data << 9.9f, 19.8f; // Expected values

    EXPECT_TRUE(t->data.isApprox(expected_data, 1e-5));

    delete t; // Clean up
}

TEST_F(OptimizerTest, SGDMomentumStep)
{
    Tensor *t = new Tensor(1, 1);
    t->data << 5.0f;
    t->grad << 2.0f;

    std::vector<Tensor *> params = {t};
    float lr = 0.1f;
    float momentum = 0.9f;
    SGD optimizer(params, lr, momentum);

    optimizer.step();
    EXPECT_NEAR(t->data(0, 0), 4.8f, 1e-5);

    t->grad << 1.0f;
    optimizer.step();
    EXPECT_NEAR(t->data(0, 0), 4.52f, 1e-5);

    delete t;
}

TEST_F(OptimizerTest, AdamStep)
{
    Tensor *t = new Tensor(1, 1);
    t->data << 1.0f;
    t->grad << 0.1f;

    std::vector<Tensor *> params = {t};
    Adam optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f);

    optimizer.step();

    EXPECT_LT(t->data(0, 0), 1.0f);

    delete t;
}

TEST_F(OptimizerTest, RMSPropStep)
{
    Tensor *t = new Tensor(1, 1);
    t->data << 2.0f;
    t->grad << 0.4f;

    std::vector<Tensor *> params = {t};
    RMSProp optimizer(params, 0.01f, 0.99f, 1e-8f, 0.0f);

    optimizer.step();

    EXPECT_LT(t->data(0, 0), 2.0f);

    delete t;
}

TEST_F(OptimizerTest, AdagradStep)
{
    Tensor *t = new Tensor(1, 1);
    t->data << 3.0f;
    t->grad << 0.3f;

    std::vector<Tensor *> params = {t};
    Adagrad optimizer(params, 0.01f, 1e-8f);

    optimizer.step();

    EXPECT_LT(t->data(0, 0), 3.0f);

    delete t;
}
