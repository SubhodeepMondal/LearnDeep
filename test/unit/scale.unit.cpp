#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eager_MatrixScale_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_scale[] = {0.23408757, 0.59419644, 0.45749621, 0.37416155,
                              0.09751165, 0.09749658, 0.03630226, 0.54136009,
                              0.37569688, 0.44254536, 0.01286531, 0.60619366,
                              0.52027665, 0.13271194, 0.1136406,  0.11462782};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);

  C = A.scale(0.625);

  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(C.getData()[i], c_scale[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_MatrixScale_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_scale[] = {0.23408757, 0.59419644, 0.45749621, 0.37416155,
                              0.09751165, 0.09749658, 0.03630226, 0.54136009,
                              0.37569688, 0.44254536, 0.01286531, 0.60619366,
                              0.52027665, 0.13271194, 0.1136406,  0.11462782};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  {
    tf::graph_context ctx;

    C = A.scale(0.625);

    ctx.run();

    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(C.getData()[i], c_scale[i], 0.0001);
    }
  }
}

TEST_F(MathTest, Scale_AutoGrad_Test_1) {
  tf::tensor A, B, C, D, E;
  A.tf_create(tf_float64, 12, 8);
  C.tf_create(tf_float64, 12, 8);

  A.tensor_of(load_bin("test/data/ScaleMulPow_Test_1_A.bin", 12 * 8).data());
  C.tensor_of(load_bin("test/data/ScaleMulPow_Test_1_C.bin", 12 * 8).data());

  {
    tf::graph_context ctx;
    B = A.scale(2.78);
    D = B.mul(C);
    E = B.pow(3);

    ctx.run();
    ctx.initialize_gradient();
    ctx.compute_gradient();

    tf::tensor A_grad = ctx.get_gradient(A);
    tf::tensor B_grad = ctx.get_gradient(B);

    auto a_grad = load_bin("test/data/ScaleMulPow_Test_1_grad_A.bin", 12 * 8);

    auto b_grad = load_bin("test/data/ScaleMulPow_Test_1_grad_B.bin", 12 * 8);

    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 12; i++) {
        EXPECT_NEAR(A_grad.getData()[i + j * 12], a_grad[i + j * 12], 1e-6);
      }
    }

    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 12; i++) {
        EXPECT_NEAR(B_grad.getData()[i + j * 12], b_grad[i + j * 12], 1e-6);
      }
    }
  }
}
