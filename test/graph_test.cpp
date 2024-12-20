/* g++ -c lib/Math/CPULibrary.cpp  -I include -o build/obj/CPULibrary.o && g++
 * test/graph_test.cpp build/obj/CPULibrary.o -lcpu -o build/exe/graph_test.out
 * -I include -L build/archieve -fopenmp  && sudo perf stat
 * ./build/exe/graph_test.out */

#include "MathLibrary.h"
#include <iostream>

int main() {
  NDMath<double, 0> A, B, C, D, E, F, G, H, I;

  A = NDMath<double, 0>(3, 2, 3, 4);
  B = NDMath<double, 0>(2, 3, 4);
  C = NDMath<double, 0>(4, 3, 3);
  D = NDMath<double, 0>(3, 4, 3);
  E = NDMath<double, 0>(2, 4, 3);
  F = NDMath<double, 0>(2, 3, 3);

  bool flag;

  A.initRandData(1, 2);
  B.initRandData(-150,-50);
  C.initRandData(2, 4);
  D.initRandData(2, 3);
  E.initRandData(5, 10);

  // B=A;
  // A.printData();
  // std::cout << "\n";
  // B.printData();

  Graph<double, 0> g;

  C = A.reducesum(g, 0);
  D = E.matmul(C, g);
  F = B.add(D, g);
  // G = F.matmul(C, g);
  // H = G.pow(2, g);
  // I = H.reducesum(g, 0, 2);

  g.optimize();
  g.execute();
  g.traversenode();
}