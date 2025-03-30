/* g++ -c core/Math/CPULibrary.cpp  -I include -o build/obj/CPULibrary.o && g++
 * test/graph_test.cpp build/obj/CPULibrary.o -lcpu -obuild/exe/graph_test.out
 * -I include -L build/archieve -fopenmp && sudo perf stat
 * ./build/exe/graph_test.out */

// #include "../core/framework/MathLibrary.h"
// #include "../core/graph/Graph.h"
#include "../api/C_API.h"
#include <iostream>

int main() {
  tensor<double> A, B, C, D, E, F, G, H, I, J;

  unsigned arr[] = {2, 3, 3};
  A = tf::tf_create(3, 4, 3);
  B = tf::tf_create(3, 3, 3);
  C = tf::tf_create(3, 4, 3);
  // D = tensor<double>(3, 3, 3);
  // E = tensor<double>(3, 4, 3);
  // F = tensor<double>(2, 3, 3);

  // bool flag;

  A.initRandData(5, 10);
  B.initRandData(2, 7.5);
  C.initRandData(-100, -50);
  // D.initRandData(2, 3);
  // E.initRandData(5, 10);

  graph g;
  // std::cout << "printing A:\n";
  // A.printData();

  // std::cout << "printing B:\n";
  // B.printData();
  // E = D.matmul(g, C);
  // F = B.add(g, E);
  // H = F.scale(g, 0.1);
  // I = H.pow(g, 3);
  // J = I.mean(g, 1);
  // tf::reducesum(g, D, A, 0,1);
  tf::matmul(g, D, A, B);
  tf::add(g, E, D, C);
  tf::pow(g, F, E, 2);
  tf::reducesum(g, G, F, 0);
  tf::mean(g, H, G, 0);
  tf::scale(g, I, G, -1.75);

  // D = A.matmul(g, B);
  // E = D.add(g, C);
  // F = E.pow(g, 2);
  // G = F.reducesum(g, 1);

  g.optimize();
  g.execute();
  g.traversenode();
  D.printDimensions();
}
