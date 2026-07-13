#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Relu_Test_1) {

  tf::tensor A, C;
  A.tf_create(tf_float64, 63, 32);

  A.tensor_of(
      load_bin("test/data/Relu_Test_1_input_sample.bin", 63 * 32).data());

  C = A.relu();

  auto output = load_bin("test/data/Relu_Test_1_output_sample.bin", 63 * 32);
  for (int i = 0; i < 63 * 32; i++) {
    EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at: " << i;
  }
}

TEST_F(MathTest, Relu_Test_2) {

  tf::tensor A, C;
  A.tf_create(tf_float64, 51, 48, 36, 72);

  A.tensor_of(
      load_bin("test/data/Relu_Test_2_input_sample.bin", 51 * 48 * 36 * 72)
          .data());

  auto output =
      load_bin("test/data/Relu_Test_2_output_sample.bin", 51 * 48 * 36 * 72);
  {
    tf::graph_context ctx;

    C = A.relu();

    ctx.run();

    auto output =
        load_bin("test/data/Relu_Test_2_output_sample.bin", 51 * 48 * 36 * 72);
    for (int i = 0; i < 51 * 48 * 36; i++) {
      EXPECT_NEAR(C.getData()[i], output[i], 1e-6);
    }
  }
}

TEST_F(MathTest, Relu_Test_3) {

  tf::tensor A, A_grad, B, B_grad, C, C_grad, D, D_grad, E, E_grad, F, G;
  A.tf_create(tf_float64, 18, 32);
  B.tf_create(tf_float64, 18, 32);
  D.tf_create(tf_float64, 34, 18);

  auto A_input = load_bin("test/data/Relu_Test_3_A_input.bin", 32 * 18);
  auto B_input = load_bin("test/data/Relu_Test_3_B_input.bin", 32 * 18);
  auto D_input = load_bin("test/data/Relu_Test_3_D_input.bin", 18 * 34);

  auto C_output = load_bin("test/data/Relu_Test_3_C_output.bin", 32 * 18);
  auto E_output = load_bin("test/data/Relu_Test_3_E_output.bin", 32 * 18);
  auto F_output = load_bin("test/data/Relu_Test_3_F_output.bin", 32 * 34);
  auto G_output = load_bin("test/data/Relu_Test_3_G_output.bin", 32 * 18);

  auto A_grad_output = load_bin("test/data/Relu_Test_3_A_grad.bin", 32 * 18);
  auto B_grad_output = load_bin("test/data/Relu_Test_3_B_grad.bin", 32 * 18);
  auto C_grad_output = load_bin("test/data/Relu_Test_3_C_grad.bin", 32 * 18);
  auto D_grad_output = load_bin("test/data/Relu_Test_3_D_grad.bin", 18 * 34);
  auto E_grad_output = load_bin("test/data/Relu_Test_3_E_grad.bin", 32 * 18);

  A.tensor_of(A_input.data());
  B.tensor_of(B_input.data());
  D.tensor_of(D_input.data());
  {
    tf::graph_context ctx;

    C = A.sub(B);
    E = C.relu();
    F = E.matmul(D);
    G = E.pow(3);

    ctx.run();

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(C.getData()[i], C_output[i], 1e-6) << "C at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(E.getData()[i], E_output[i], 1e-6) << "E at: " << i;

    for (int i = 0; i < 32 * 34; i++)
      EXPECT_NEAR(F.getData()[i], F_output[i], 1e-6) << "F at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(G.getData()[i], G_output[i], 1e-6) << "G at: " << i;

    ctx.initialize_gradient();
    ctx.compute_gradient();

    A_grad = ctx.get_gradient(A);
    B_grad = ctx.get_gradient(B);
    C_grad = ctx.get_gradient(C);
    D_grad = ctx.get_gradient(D);
    E_grad = ctx.get_gradient(E);

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(A_grad.getData()[i], A_grad_output[i], 1e-6)
          << "A_grad at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(B_grad.getData()[i], B_grad_output[i], 1e-6)
          << "B_grad at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(C_grad.getData()[i], C_grad_output[i], 1e-6)
          << "C_grad at: " << i;

    for (int i = 0; i < 18 * 34; i++)
      EXPECT_NEAR(D_grad.getData()[i], D_grad_output[i], 1e-6)
          << "D_grad at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(E_grad.getData()[i], E_grad_output[i], 1e-6)
          << "E_grad at: " << i;
  }
}
