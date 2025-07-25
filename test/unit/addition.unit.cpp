#include <gtest/gtest.h>
#include <tensor.h>
#include "LinearAlgebraFixtures.unit.hpp"


TEST_F(MathTest, MatrixAddition_2D) {

  std::float64_t c_add[] = {0.50509996, 0.79524308, 0.82034027, 0.83129132,
                      1.12101639, 0.65899134, 1.4396007,  0.83401931,
                      0.58314816, 0.89092581, 1.34648869, 0.76883297,
                      1.1066349,  1.07983453, 1.3015432,  1.21275137};

  tf::add(g, C, A, B);
  tf::graph_optimize(g);
  tf::graph_execute(g);

  auto *tensorC_add = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_add->getData()[i], c_add[i], 0.0001);
  }
}
