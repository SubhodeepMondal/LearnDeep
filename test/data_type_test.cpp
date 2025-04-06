#include "../api/C_API.h"
#include <iostream>

int main() {
  tensor<double> A, B, C, D, E, F, G, H, I, J;

  unsigned arr[] = {2, 3, 3};
  A = tf::tf_create(3, 4, 3);
  B = tf::tf_create(3, 3, 3);
  C = tf::tf_create(3, 4, 3);

  tf::tensor_of(A, 5, 10);
  tf::tensor_of(B, 2, 7.5);
  tf::tensor_of(C, -100, -50);
}
