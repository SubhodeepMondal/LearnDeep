#include <iostream>
#include "MathLibrary.h"

int main()
{
    NDMath<double, 0> A, B, C, D, E, F, G;

    A = NDMath<double, 0>(2, 3, 3);
    B = NDMath<double, 0>(2, 3, 3);
    C = NDMath<double, 0>(4, 3, 3);
    D = NDMath<double, 0>(3, 3, 3);
    E = NDMath<double, 0>(2, 3, 3);
    F = NDMath<double, 0>(2, 3, 3);

    bool flag;

    A.initRandData(1, 2);
    B.initRandData(-5, 2);
    C.initRandData(2, 4);
    D.initRandData(2, 3);
    E.initRandData(-5, 2);


    Graph<double, 0> g;

    C = A.mul(B, g);
    D = E.mul(C, g);
    F = C.mul(D, g);
    G = F.mul(C, g);


    g.optimize();
    g.execute();
    g.traversenode();
}