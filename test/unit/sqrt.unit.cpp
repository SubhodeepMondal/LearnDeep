#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eager_MatrixSQRT_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_scale[] = {0.61199683, 0.9750458,  0.85556645, 0.7737302,
                              0.39499195, 0.39496142, 0.24100542, 0.93068585,
                              0.77531607, 0.84147049, 0.14347297, 0.98484001,
                              0.91238295, 0.46080268, 0.42640939, 0.42825753};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  // C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  C = A.sqrt();

  auto *tensorC_sqrt = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_sqrt->getData()[i], c_scale[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_MatrixSQRT_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_scale[] = {0.61199683, 0.9750458,  0.85556645, 0.7737302,
                              0.39499195, 0.39496142, 0.24100542, 0.93068585,
                              0.77531607, 0.84147049, 0.14347297, 0.98484001,
                              0.91238295, 0.46080268, 0.42640939, 0.42825753};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  {
    tf::graph_context ctx;

    C = A.sqrt();

    ctx.run();

    auto *tensorC_sqrt = static_cast<Tensor<std::float64_t> *>(C.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorC_sqrt->getData()[i], c_scale[i], 0.0001);
    }
  }
}
