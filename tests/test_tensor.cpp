#include <gtest/gtest.h>
#include "../include/tensor.hpp"
#include "../include/linear.hpp"

TEST(TensorTest, Creation)
{
    Tensor A(2, 3);
    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_EQ(A(0, 0), 0.0f);
}

TEST(TensorTest, Addition)
{
    Tensor A(2, 2);
    Tensor B(2, 2);
    A(0, 0) = 1.0f;
    A(0, 1) = 2.0f;
    A(1, 0) = 3.0f;
    A(1, 1) = 4.0f;

    B(0, 0) = 5.0f;
    B(0, 1) = 6.0f;
    B(1, 0) = 7.0f;
    B(1, 1) = 8.0f;

    Tensor *C = A + &B;
    EXPECT_EQ(C->data(0, 0), 6.0f);
    EXPECT_EQ(C->data(0, 1), 8.0f);
    EXPECT_EQ(C->data(1, 0), 10.0f);
    EXPECT_EQ(C->data(1, 1), 12.0f);

    delete C;
}

TEST(TensorTest, Initialization)
{
    Tensor A(2, 2, "ones");
    EXPECT_EQ(A(0, 0), 1.0f);
    EXPECT_EQ(A(0, 1), 1.0f);
    EXPECT_EQ(A(1, 0), 1.0f);
    EXPECT_EQ(A(1, 1), 1.0f);

    Tensor B(2, 2, "zeros");
    EXPECT_EQ(B(0, 0), 0.0f);
    EXPECT_EQ(B(0, 1), 0.0f);
    EXPECT_EQ(B(1, 0), 0.0f);
    EXPECT_EQ(B(1, 1), 0.0f);

    Tensor C(2, 2, "random");
    // Just check that the values are within the expected range
    for (int i = 0; i < C.rows(); ++i)
    {
        for (int j = 0; j < C.cols(); ++j)
        {
            EXPECT_GE(C(i, j), -1.0f);
            EXPECT_LE(C(i, j), 1.0f);
        }
    }
}

TEST(LinearTest, ForwardPass)
{
    linear layer(3, 2); // Input size 2, output size 3
    Tensor input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Tensor *output = layer.forward(&input);
    EXPECT_EQ(output->rows(), 1);
    EXPECT_EQ(output->cols(), 2);

    delete output;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
