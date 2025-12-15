#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eager_MatrixMean_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_mean[] = {0.4910291, 0.50678013, 0.24812175, 0.65453725};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  C = A.mean(0);

  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(C.getData()[i], c_mean[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_MatrixMean_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_mean[] = {0.4910291, 0.50678013, 0.24812175, 0.65453725};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  {
    tf::graph_context ctx;

    C = A.mean(0);

    ctx.run();

    for (int i = 0; i < 4; i++) {
      EXPECT_NEAR(C.getData()[i], c_mean[i], 0.0001);
    }
  }
}
