#include "../api/tensor.h"
#include <iostream>

int main() {
  tf::tensor A, B, C, D, E, F, G, H, I, J;
  unsigned n = 1 << 2;
  std::cout << n << "\n";
  tf::tf_create(A, tf_float64, n, n);
  tf::tf_create(B, tf_float64, n, n);
  tf::tf_create(C, tf_float64, 3, 4, 3);
  tf::tf_create(D, tf_float64, n, n);
  tf::tf_create(E, tf_float64, 3, 3, 2);
  tf::tf_create(F, tf_float64, 3, 3, 2);
  tf::tf_create(G, tf_float64, 3, 3, 2);
  tf::tf_create(H, tf_float64, 3, 3, 2);
  tf::tf_create(I, tf_float64, 3, 3, 2);
  tf::tf_create(J, tf_float64, 3, 3, 2);

  tf::tensor_of(A, 5, 10);
  tf::tensor_of(B, 2, 7.5);
  tf::tensor_of(C, -100, 50);
  tf::tensor_of(D, -100, -50);

  tf::graph g;
  tf::tf_create_graph(g);

  tf::matmul(g, C, A, B);
  // tf::add(g, E, D, C);
  // tf::scale(g, C, B, 0.15);
  // tf::add(g, F, D, E);
  // tf::scale(g, G, F, 0.15);
  // tf::pow(g, H, G, 2);
  // tf::reducesum(g, B, A, 0);
  // tf::mean(g, C, B, 0);

  // g.optimize();
  // g.execute();
  // g.traversenode();

  tf::graph_optimize(g);
  tf::graph_execute(g);
  tf::graph_travarse_node(g);
  // std::cout << "\n";
  // A.printDimensions();
}
