#include <iostream>
#include "MathLibrary.h"

// template <typename T, int typeFlag>
// class NDMath: public NDArray<T,typeFlag>
// {
//     public:
//     NDMath(unsigned n, unsigned... args): NDArray<T,typeFlag>(n, arr)
//     {

//     }
// };
// template <typename T, int typeFlag>
NDMath<double, 0> returnNDMath()
{
    unsigned a[] = {2, 4, 3};

    NDMath<double, 0> output;
    output = NDMath<double, 0>(3, a);

    return output;
}

int main()
{
    unsigned a[] = {3, 4, 3};

    NDMath<double, 0> A, B, C, D, E, F; //;
    unsigned m = 1 << 10;
    unsigned n = 1 << 14;
    unsigned k = 1 << 10;
    std::cout << m << " " << n << " " << k << "\n";
    A = NDMath<double, 0>(m, n);
    B = NDMath<double, 0>(n, k); // 3, 2, 2, 3
    C = NDMath<double, 0>(2, 4, 3);
    D = NDMath<double, 0>(2, 4, 3);
    E = NDMath<double, 0>(2, 4);

    // A = returnNDMath();
    // A=A;

    A.initRandData(-1, 1);
    A.printDimensions();
    std::cout << "\n";
    // A.printData();

    std::cout << std::endl;
    B.initRandData(-1, 1);
    B.printDimensions();
    std::cout << "\n";
    // B.printData();

    // C = A;

    // C.printData();

    // C.initRandData(-5, 5);
    // C.printDimensions();
    // std::cout << "\n";
    // C.printData();

    // D.initRandData(-8, 12);
    // D.printDimensions();
    // std::cout << "\n";
    // D.printData();

    E = A.matrixMultiplication(B);
    E.printDimensions();
    std::cout << "\n";
    // E.printData();

    // F = E - D;
    // F.printDimensions();
    // std::cout << "\n";
    // F.printData();
}