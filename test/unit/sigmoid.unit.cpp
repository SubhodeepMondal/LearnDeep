#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eager_Matrixsigmoid_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_sigmoid[] = {0.59255557, 0.72125881, 0.67524268, 0.64534933,
                                0.53892573, 0.53891974, 0.51451682, 0.70394941,
                                0.64591136, 0.66997513, 0.50514594, 0.72510153,
                                0.69687117, 0.55288622, 0.54533142, 0.54572303};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  // C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  C = A.sigmoid();

  auto *tensorC_sigmoid = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_sigmoid->getData()[i], c_sigmoid[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_Matrixsigmoid_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_sigmoid[] = {0.59255557, 0.72125881, 0.67524268, 0.64534933,
                                0.53892573, 0.53891974, 0.51451682, 0.70394941,
                                0.64591136, 0.66997513, 0.50514594, 0.72510153,
                                0.69687117, 0.55288622, 0.54533142, 0.54572303};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  tf::graph g_sigmoid;
  g_sigmoid.tf_create_graph();

  C = A.sigmoid(g_sigmoid);

  g_sigmoid.graph_execute();

  auto *tensorC_sigmoid = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_sigmoid->getData()[i], c_sigmoid[i], 0.0001);
  }
  g_sigmoid.graph_clear();
}
