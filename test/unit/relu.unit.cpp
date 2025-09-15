#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eager_Matrixrelu_2D) {

  std::float64_t a[] = {-0.12545988, 0.45071431,  0.23199394,  0.09865848,
                        -0.34398136, -0.34400548, -0.44191639, 0.36617615,
                        0.10111501,  0.20807258,  -0.47941551, 0.46990985,
                        0.33244264,  -0.28766089, -0.31817503, -0.31659549};

  std::float64_t c_relu[] = {0.0,        0.45071431, 0.23199394, 0.09865848,
                             0.0,        0.0,        0.0,        0.36617615,
                             0.10111501, 0.20807258, 0.0,        0.46990985,
                             0.33244264, 0.0,        0.0,        0.0};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  // C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  C = A.relu();

  auto *tensorC_relu = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_relu->getData()[i], c_relu[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_Matrixrelu_2D) {

  std::float64_t a[] = {-0.12545988, 0.45071431,  0.23199394,  0.09865848,
                        -0.34398136, -0.34400548, -0.44191639, 0.36617615,
                        0.10111501,  0.20807258,  -0.47941551, 0.46990985,
                        0.33244264,  -0.28766089, -0.31817503, -0.31659549};

  std::float64_t c_relu[] = {0.0,        0.45071431, 0.23199394, 0.09865848,
                             0.0,        0.0,        0.0,        0.36617615,
                             0.10111501, 0.20807258, 0.0,        0.46990985,
                             0.33244264, 0.0,        0.0,        0.0};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  tf::graph g_scale;
  g_scale.tf_create_graph();

  C = A.relu(g_scale);

  g_scale.graph_execute();

  auto *tensorC_relu = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_relu->getData()[i], c_relu[i], 0.0001);
  }
  g_scale.graph_clear();
}
