#include "tensor.hpp"
#include <iostream>

int main() {
    // Create two 2x2 matrices
    Tensor A(2, 2);
    Tensor B(2, 2);

    // Fill A with 1.0, B with 2.0
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;

    std::cout << "Matrix A:\n"; A.print();
    std::cout << "Matrix B:\n"; B.print();

    // Operator+ expects Tensor* and returns Tensor*
    // We pass the address of B (&B).
    Tensor* C = A + &B;
    
    Tensor D = Tensor(2, 2, "ones");
    Tensor* E = A.transpose();

    std::cout << "Result of A + B:\n";
    C->print(); // Access member via pointer arrow ->

    std::cout << "The zero matrix is\n";
    D.print();

    std::cout << "The transpose of A is\n";
    E->print();

    // Clean up heap allocated memory returned by operator+
    delete C;
    delete E;

    return 0;
}