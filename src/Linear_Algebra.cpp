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

int main()
{
    // NDArray<double, 0> A(2,a);
    // A.printDimensions();

    // A.initData(3.0);
    // A.printData();

    NDMath<double, 0> A, B, C;
    A = NDMath<double, 0>(2, 3);
    B = NDMath<double, 0>(4, 2);

    A.initData(3.0);
    A.printData();

    std::cout << std::endl;

    B.initData(2.4);
    B.printData();

    std::cout << std::endl;

    C = A.matrixMultiplication(B);
    C.printData();
}