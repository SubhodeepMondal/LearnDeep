#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, MatrixMultiplication_2D) {

  std::float64_t c_mul[] = {0.00400337, 0.05670912, 0.15322661, 0.16799542,
                            0.31416848, 0.07367846, 0.50907246, 0.16428851,
                            0.08410975, 0.0861962,  0.39687732, 0.05894089,
                            0.17593129, 0.19838859, 0.38456974, 0.36047908};
                            
  tf::mul(g, C, A, B);
  tf::graph_optimize(g);
  tf::graph_execute(g);

  auto *tensorC_mul = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_mul->getData()[i], c_mul[i], 0.0001);
  }
}
