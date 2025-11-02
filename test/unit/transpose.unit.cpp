#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eager_MatrixTranspose_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_t[] = {0.37454012, 0.15601864, 0.60111501, 0.83244264,
                          0.95071431, 0.15599452, 0.70807258, 0.21233911,
                          0.73199394, 0.05808361, 0.02058449, 0.18182497,
                          0.59865848, 0.86617615, 0.96990985, 0.18340451};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  C = A.transpose();

  auto *tensorC_transpose = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_transpose->getData()[i], c_t[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_MatrixTranspose_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_t[] = {0.37454012, 0.15601864, 0.60111501, 0.83244264,
                          0.95071431, 0.15599452, 0.70807258, 0.21233911,
                          0.73199394, 0.05808361, 0.02058449, 0.18182497,
                          0.59865848, 0.86617615, 0.96990985, 0.18340451};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  tf::graph g_transpose;
  g_transpose.tf_create_graph();

  C = A.transpose(g_transpose);

  g_transpose.graph_execute();

  auto *tensorC_transpose = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_transpose->getData()[i], c_t[i], 0.0001);
  }
  g_transpose.graph_clear();
}
