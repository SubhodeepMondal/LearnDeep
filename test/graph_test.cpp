/* g++ -c core/Math/CPULibrary.cpp  -I include -o build/obj/CPULibrary.o && g++
 * test/graph_test.cpp build/obj/CPULibrary.o -lcpu -obuild/exe/graph_test.out
 * -I include -L build/archieve -fopenmp && sudo perf stat
 * ./build/exe/graph_test.out */

#include "MathLibrary.h"
#include <iostream>

int main() {
  NDMath<double, 0> A, B, C, D, E, F, G, H, I;

  A = NDMath<double, 0>(3, 2, 3, 4);
  B = NDMath<double, 0>(2, 3, 4);
  C = NDMath<double, 0>(4, 8, 3);
  D = NDMath<double, 0>(3, 4, 3);
  E = NDMath<double, 0>(2, 4, 3);
  F = NDMath<double, 0>(2, 3, 3);

  bool flag;

  A.initRandData(-5, 10);
  B.initRandData(-150, -50);
  C.initRandData(2, 4);
  D.initRandData(2, 3);
  E.initRandData(5, 10);

  Graph<double, 0> g;

  C = A.mean(g, 2);
  // D = C.scale(g, 1.0f / A.getDimensions()[0]);
  // D = E.matmul(g, C);
  // E = B.add(g, D);
  // F = E.scale(g, 0.015);
  // H = F.pow(g, 2);

  g.optimize();
  g.execute();
  g.traversenode();
}
