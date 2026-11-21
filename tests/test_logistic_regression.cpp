#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/logistic_regression.hpp"
#include "../include/tensor.hpp"

class LogisticRegressionTest : public ::testing::Test
{
};

// ---------- Initialization ----------
TEST_F(LogisticRegressionTest, BinaryInitializationTest)
{
    LogisticRegression model(3, 1, 0.01f);
    EXPECT_EQ(model.get_weights()->rows(), 3);
    EXPECT_EQ(model.get_weights()->cols(), 1);
    EXPECT_EQ(model.get_bias()->rows(), 1);
    EXPECT_EQ(model.get_bias()->cols(), 1);
}

TEST_F(LogisticRegressionTest, MulticlassInitializationTest)
{
    LogisticRegression model(4, 3, 0.01f);
    EXPECT_EQ(model.get_weights()->rows(), 4);
    EXPECT_EQ(model.get_weights()->cols(), 3);
    EXPECT_EQ(model.get_bias()->cols(), 3);
}

TEST_F(LogisticRegressionTest, InvalidOptimizerThrows)
{
    EXPECT_THROW(LogisticRegression(2, 1, 0.01f, "RMSprop"), std::invalid_argument);
}

// ---------- predict_proba properties ----------
TEST_F(LogisticRegressionTest, BinaryProbabilitiesInRange)
{
    LogisticRegression model(2, 1);
    Tensor X(20, 2);
    X.data.setRandom();

    Tensor *p = model.predict_proba(&X);
    ASSERT_EQ(p->rows(), 20);
    ASSERT_EQ(p->cols(), 1);
    for (int i = 0; i < p->rows(); ++i)
    {
        EXPECT_GE((*p)(i, 0), 0.0f);
        EXPECT_LE((*p)(i, 0), 1.0f);
    }
    delete p;
}

TEST_F(LogisticRegressionTest, SoftmaxRowsSumToOne)
{
    LogisticRegression model(3, 4);
    Tensor X(15, 3);
    X.data.setRandom();

    Tensor *p = model.predict_proba(&X);
    ASSERT_EQ(p->cols(), 4);
    for (int i = 0; i < p->rows(); ++i)
    {
        EXPECT_NEAR(p->data.row(i).sum(), 1.0f, 1e-5f);
        EXPECT_GE(p->data.row(i).minCoeff(), 0.0f);
    }
    delete p;
}

TEST_F(LogisticRegressionTest, PredictOutputsValidLabels)
{
    LogisticRegression model(2, 1);
    Tensor X(20, 2);
    X.data.setRandom();
    Tensor *pred = model.predict(&X);
    for (int i = 0; i < pred->rows(); ++i)
    {
        float v = (*pred)(i, 0);
        EXPECT_TRUE(v == 0.0f || v == 1.0f);
    }
    delete pred;
}

// ---------- Convergence ----------
TEST_F(LogisticRegressionTest, BinaryConvergenceTest)
{
    // Linearly separable: label = 1 if x0 + x1 > 0
    int n = 200;
    Tensor X(n, 2), y(n, 1);
    X.data.setRandom();
    for (int i = 0; i < n; ++i)
        y(i, 0) = (X(i, 0) + X(i, 1) > 0.0f) ? 1.0f : 0.0f;

    LogisticRegression model(2, 1, 0.5f);
    float final_loss = model.fit(&X, &y, 500, 1000);
    EXPECT_LT(final_loss, 0.2f);

    Tensor *pred = model.predict(&X);
    int correct = 0;
    for (int i = 0; i < n; ++i)
        if ((*pred)(i, 0) == y(i, 0))
            correct++;
    EXPECT_GT(static_cast<float>(correct) / n, 0.95f);
    delete pred;
}

TEST_F(LogisticRegressionTest, MulticlassConvergenceTest)
{
    // 3 classes decided by which of 3 features is largest.
    // cross_entropy_loss expects ONE-HOT labels with the SAME shape as the
    // model's output: y is (n, 3), not (n, 1) class indices.
    int n = 300;
    Tensor X(n, 3), y(n, 3);
    X.data.setRandom();
    y.data.setZero();
    for (int i = 0; i < n; ++i)
    {
        int idx;
        X.data.row(i).maxCoeff(&idx);
        y(i, idx) = 1.0f; // one-hot
    }

    LogisticRegression model(3, 3, 0.5f, "Adam");
    model.fit(&X, &y, 500, 1000);

    Tensor *pred = model.predict(&X);
    int correct = 0;
    for (int i = 0; i < n; ++i)
    {
        // pred holds a class INDEX; y is one-hot, so recover the true class
        // by taking the argmax of the row before comparing.
        int true_class;
        y.data.row(i).maxCoeff(&true_class);
        if (static_cast<int>((*pred)(i, 0)) == true_class)
            correct++;
    }
    EXPECT_GT(static_cast<float>(correct) / n, 0.9f);
    delete pred;
}

TEST_F(LogisticRegressionTest, AdamConvergesBinary)
{
    int n = 100;
    Tensor X(n, 1), y(n, 1);
    X.data.setRandom();
    for (int i = 0; i < n; ++i)
        y(i, 0) = (X(i, 0) > 0.0f) ? 1.0f : 0.0f;

    LogisticRegression model(1, 1, 0.1f, "Adam");
    EXPECT_LT(model.fit(&X, &y, 300, 1000), 0.3f);
}

// ---------- Loss behavior ----------
TEST_F(LogisticRegressionTest, LossDecreases)
{
    int n = 100;
    Tensor X(n, 2), y(n, 1);
    X.data.setRandom();
    for (int i = 0; i < n; ++i)
        y(i, 0) = (X(i, 0) > X(i, 1)) ? 1.0f : 0.0f;

    LogisticRegression model(2, 1, 0.1f);
    float loss_early = model.fit(&X, &y, 5, 1000);
    float loss_late = model.fit(&X, &y, 200, 1000);
    EXPECT_LT(loss_late, loss_early);
}

TEST_F(LogisticRegressionTest, InitialLossNearLn2)
{
    // With ~zero weights, binary CE should be about ln(2) = 0.693
    int n = 500;
    Tensor X(n, 2), y(n, 1);
    X.data.setRandom();
    for (int i = 0; i < n; ++i)
        y(i, 0) = (i % 2 == 0) ? 1.0f : 0.0f;

    LogisticRegression model(2, 1, 0.0f); // lr = 0, no learning
    float loss = model.fit(&X, &y, 1, 1000);
    EXPECT_NEAR(loss, 0.693f, 0.1f);
}

// ---------- Memory ----------
TEST_F(LogisticRegressionTest, BinaryLoopMemoryAccumulationTest)
{
    long long initial = Tensor::active_tensors;
    {
        LogisticRegression model(2, 1, 0.01f);
        Tensor X(10, 2), y(10, 1);
        X.data.setRandom();
        y.data.setZero();
        model.fit(&X, &y, 50, 1000);

        Tensor *p = model.predict_proba(&X);
        delete p;
        Tensor *c = model.predict(&X);
        delete c;
    }
    EXPECT_EQ(Tensor::active_tensors, initial)
        << "Leaked tensors: " << (Tensor::active_tensors - initial);
}

TEST_F(LogisticRegressionTest, MulticlassLoopMemoryAccumulationTest)
{
    long long initial = Tensor::active_tensors;
    {
        LogisticRegression model(3, 4, 0.01f);
        Tensor X(10, 3), y(10, 4); // y shape MUST match model output (one-hot)
        X.data.setRandom();
        y.data.setZero();
        model.fit(&X, &y, 50, 1000);
    }
    EXPECT_EQ(Tensor::active_tensors, initial);
}