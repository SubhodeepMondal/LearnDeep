#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

namespace {

void expect_near_tensor(const tf::tensor &actual,
                        const std::vector<std::float64_t> &expected) {
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(actual.getData()[i], expected[i], 1e-6) << "at: " << i;
  }
}

} // namespace

TEST_F(MathTest, Logarithm_Test_1) {
  tf::tensor A, C;
  A.tf_create(tf_float64, 18, 67);

  A.tensor_of(load_bin("test/data/MatrixLog_Test_1_input.bin", 67 * 18).data());
  auto c = load_bin("test/data/MatrixLog_Test_1_output.bin", 67 * 18);

  C = A.log();
  for (int i = 0; i < 18 * 67; i++) {
    EXPECT_NEAR(C.getData()[i], c[i], 1e-6) << "at: " << i;
  }
}

TEST_F(MathTest, Logarithm_Test_2) {
  tf::tensor A, B, C;
  A.tf_create(tf_float64, 13, 21, 10, 8, 9);

  A.tensor_of(
      load_bin("test/data/MatrixLog_Test_2_input.bin", 9 * 8 * 10 * 21 * 13)
          .data());
  auto c =
      load_bin("test/data/MatrixLog_Test_2_output.bin", 9 * 8 * 10 * 21 * 13);
  {
    tf::graph_context ctx;
    C = A.log();

    ctx.run();

    for (int i = 0; i < 9 * 8 * 10 * 21 * 13; i++) {
      EXPECT_NEAR(C.getData()[i], c[i], 1e-6) << "at: " << i;
    }
  }
}

TEST_F(MathTest, Logarithm_AutoGrad_Test_1) {
  constexpr size_t tensor_size = 56 * 83;

  tf::tensor A, B, d_B;
  A.tf_create(tf_float64, 83, 56);
  d_B.tf_create(tf_float64, 83, 56);

  A.tensor_of(
      load_bin("test/data/MatrixLog_Autograd_Test_1_A.bin", tensor_size)
          .data());
  d_B.tensor_of(1.0, 1.0);

  {
    tf::graph_context ctx;

    B = A.log();
    ctx.inject_gradient(B, d_B);

    ctx.run();

    expect_near_tensor(
        B, load_bin("test/data/MatrixLog_Autograd_Test_1_B.bin", tensor_size));

    ctx.initialize_gradient();
    ctx.compute_gradient();

    tf::tensor A_grad = ctx.get_gradient(A);
    expect_near_tensor(
        A_grad,
        load_bin("test/data/MatrixLog_Autograd_Test_1_A_grad.bin",
                 tensor_size));
  }
}

TEST_F(MathTest, Logarithm_AutoGrad_Test_2) {
  constexpr size_t a_size = 26 * 45;
  constexpr size_t c_size = 26 * 13;
  constexpr size_t d_size = 45 * 13;

  tf::tensor A, B, C, D, E, F, d_C, d_F;
  A.tf_create(tf_float64, 45, 26);
  D.tf_create(tf_float64, 13, 45);
  E.tf_create(tf_float64, 45, 26);
  d_C.tf_create(tf_float64, 13, 26);
  d_F.tf_create(tf_float64, 45, 26);

  A.tensor_of(
      load_bin("test/data/MatrixLog_Autograd_Test_2_A.bin", a_size).data());
  D.tensor_of(
      load_bin("test/data/MatrixLog_Autograd_Test_2_D.bin", d_size).data());
  E.tensor_of(
      load_bin("test/data/MatrixLog_Autograd_Test_2_E.bin", a_size).data());
  d_C.tensor_of(1.0, 1.0);
  d_F.tensor_of(1.0, 1.0);

  {
    tf::graph_context ctx;

    B = A.log();
    C = B.matmul(D);
    F = B.mul(E);
    ctx.inject_gradient(C, d_C);
    ctx.inject_gradient(F, d_F);

    ctx.run();

    expect_near_tensor(
        B, load_bin("test/data/MatrixLog_Autograd_Test_2_B.bin", a_size));
    expect_near_tensor(
        C, load_bin("test/data/MatrixLog_Autograd_Test_2_C.bin", c_size));
    expect_near_tensor(
        F, load_bin("test/data/MatrixLog_Autograd_Test_2_F.bin", a_size));

    ctx.initialize_gradient();
    ctx.compute_gradient();

    tf::tensor A_grad = ctx.get_gradient(A);
    tf::tensor D_grad = ctx.get_gradient(D);
    tf::tensor E_grad = ctx.get_gradient(E);

    expect_near_tensor(
        A_grad,
        load_bin("test/data/MatrixLog_Autograd_Test_2_A_grad.bin", a_size));
    expect_near_tensor(
        D_grad,
        load_bin("test/data/MatrixLog_Autograd_Test_2_D_grad.bin", d_size));
    expect_near_tensor(
        E_grad,
        load_bin("test/data/MatrixLog_Autograd_Test_2_E_grad.bin", a_size));
  }
}
