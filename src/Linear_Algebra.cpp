#include <iostream>
#include "NDynamicArray.h"

int main()
{
    NDArray<double, 0> A ; //= NDArray<double, 0>(2, 3, 3);
    A = NDArray<double, 0>(2,3,3);

    A.initData(3.0);
    A.printData();
}