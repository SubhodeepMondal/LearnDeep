#include "LinearAlgebraFixtures.unit.hpp"
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "sub.data.hpp"

TEST_F(MathTest, MatrixSubtraction_Eager_Test_1) {

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 17, 15);
  B.tf_create(tf_float64, 17, 15);

  A.tensor_of(a_17_15_matrixsubtraction_eager_test_1);
  B.tensor_of(b_17_15_matrixsubtraction_eager_test_1);

  C = A.sub(B);

  for (unsigned j = 0; j < 15; j++)
    for (int i = 0; i < 17; i++) {
      unsigned index = i + j * 17;
      EXPECT_NEAR(C.getData()[index],
                  out_17_15_matrixsubtraction_eager_test_1[index], 1e-6)
          << "at : " << index << "\n";
    }
}

TEST_F(MathTest, MatrixSubtraction_Eager_Test_2) {

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 13, 7, 5, 3);
  B.tf_create(tf_float64, 13, 7, 5, 3);

  A.tensor_of(a_13_7_5_3_matrixsubtraction_eager_test_2);
  B.tensor_of(b_13_7_5_3_matrixsubtraction_eager_test_2);

  C = A.sub(B);

  for (unsigned k = 0; k < 15; k++)
    for (unsigned j = 0; j < 7; j++)
      for (unsigned i = 0; i < 13; i++) {
        unsigned index = i + j * 13 + k * 91;
        EXPECT_NEAR(C.getData()[index],
                    out_13_7_5_3_matrixsubtraction_eager_test_2[index], 1e-6)
            << "at : " << index << "\n";
      }
}

TEST_F(MathTest, MatrixSubtraction_Eager_Test_3) {

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 16, 7, 6);
  B.tf_create(tf_float64, 16, 1, 6);

  A.tensor_of(a_16_7_6_matrixsubtraction_eager_test_3);
  B.tensor_of(b_16_1_6_matrixsubtraction_eager_test_3);

  C = A.sub(B);

  for (unsigned k = 0; k < 6; k++)
    for (unsigned j = 0; j < 7; j++)
      for (unsigned i = 0; i < 16; i++) {
        unsigned index = i + j * 16 + k * 112;
        EXPECT_NEAR(C.getData()[index],
                    out_16_7_6_matrixsubtraction_eager_test_3[index], 1e-6)
            << "at : " << index << "\n";
      }
}

TEST_F(MathTest, MatrixSubtraction_Graph_Test_4) {

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 18, 19);
  B.tf_create(tf_float64, 18, 1);

  A.tensor_of(a_18_19_matrixsubtraction_eager_test_4);
  B.tensor_of(b_18_1_matrixsubtraction_eager_test_4);
  {
    tf::graph_context ctx;

    C = A.sub(B);

    ctx.run();

    for (unsigned j = 0; j < 19; j++)
      for (unsigned i = 0; i < 18; i++) {
        unsigned index = i + j * 18;
        EXPECT_NEAR(C.getData()[index],
                    output_18_19_matrixsubtraction_eager_test_4[index], 1e-6)
            << "at : " << index << "\n";
      }
  }
}

TEST_F(MathTest, MatrixSubtraction_Graph_Test_5) {

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 12, 10, 7, 6, 5);
  B.tf_create(tf_float64, 1, 10, 1, 6);

  A.tensor_of(a_12_10_7_6_5_matrixsubtraction_eager_test_5);
  B.tensor_of(b_1_10_1_6_matrixsubtraction_eager_test_5);
  {
    tf::graph_context ctx;

    C = A.sub(B);

    ctx.run();

    for (unsigned k = 0; k < 210; k++)
      for (unsigned j = 0; j < 10; j++)
        for (unsigned i = 0; i < 12; i++) {
          unsigned index = i + j * 12 + k * 120;
          EXPECT_NEAR(C.getData()[index],
                      output_12_10_7_6_5_matrixsubtraction_eager_test_5[index],
                      1e-6);
        }
  }
}

TEST_F(MathTest, MatrixSubtraction_GraphGradient_Test_6) {

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 38, 32, 5);
  B.tf_create(tf_float64, 38, 1, 5);

  A.tensor_of(a_38_32_5_matrixsubtraction_graphgradient_test_6);
  B.tensor_of(b_38_1_5_matrixsubtraction_graphgradient_test_6);
  {
    tf::graph_context ctx;

    C = A.sub(B);

    ctx.run();

    for (unsigned k = 0; k < 5; k++)
      for (unsigned j = 0; j < 32; j++)
        for (unsigned i = 0; i < 38; i++) {
          unsigned index = i + j * 38 + k * 1216;
          EXPECT_NEAR(
              C.getData()[index],
              output_38_32_5_matrixsubtraction_graphgradient_test_6[index],
              1e-6)
              << "at: " << index << "\n";
        }
    ctx.initialize_gradient();
    ctx.compute_gradient();

    tf::tensor B_grad = ctx.get_gradient(B);

    for (unsigned j = 0; j < 5; j++)
      for (unsigned i = 0; i < 38; i++) {
        unsigned index = i + j * 38;
        EXPECT_NEAR(B_grad.getData()[index], -32.0, 1e-6)
            << "at: " << index << "\n";
      }
  }
}

TEST_F(MathTest, MatrixSubtraction_GraphGradient_Test_7) {

  tf::tensor A, B, C, D, E, F, G;
  A.tf_create(tf_float64, 42, 32);
  B.tf_create(tf_float64, 42, 1);
  E.tf_create(tf_float64, 72, 42);
  F.tf_create(tf_float64, 42, 32);

  A.tensor_of(a_42_32_matrixsubtraction_graphgradient_test_7);
  B.tensor_of(b_42_1_matrixsubtraction_graphgradient_test_7);
  E.tensor_of(e_72_32_matrixsubtraction_graphgradient_test_7);
  F.tensor_of(f_42_32_matrixsubtraction_graphgradient_test_7);
  {
    tf::graph_context ctx;

    C = A.sub(B);
    D = F.mul(C);
    G = C.matmul(E);

    ctx.run();

    for (unsigned j = 0; j < 32; j++)
      for (unsigned i = 0; i < 42; i++) {
        unsigned index = i + j * 42;
        EXPECT_NEAR(C.getData()[index],
                    c_42_32_matrixsubtraction_graphgradient_test_7[index], 1e-6)
            << "at: " << index << "\n";
      }

    for (unsigned j = 0; j < 32; j++)
      for (unsigned i = 0; i < 42; i++) {
        unsigned index = i + j * 42;
        EXPECT_NEAR(D.getData()[index],
                    d_42_32_matrixsubtraction_graphgradient_test_7[index], 1e-6)
            << "at: " << index << "\n";
      }

    for (unsigned j = 0; j < 32; j++)
      for (unsigned i = 0; i < 72; i++) {
        unsigned index = i + j * 72;
        EXPECT_NEAR(G.getData()[index],
                    f_72_32_matrixsubtraction_graphgradient_test_7[index], 1e-6)
            << "at: " << index << "\n";
      }
    ctx.initialize_gradient();
    ctx.compute_gradient();

    tf::tensor A_grad = ctx.get_gradient(A);
    tf::tensor B_grad = ctx.get_gradient(B);
    tf::tensor C_grad = ctx.get_gradient(C);
    tf::tensor E_grad = ctx.get_gradient(E);
    tf::tensor F_grad = ctx.get_gradient(F);

    for (unsigned j = 0; j < 32; j++)
      for (unsigned i = 0; i < 42; i++) {
        unsigned index = i + j * 42;
        EXPECT_NEAR(A_grad.getData()[index],
                    a_42_32_grad_matrixsubtraction_graphgradient_test_7[index],
                    1e-6)
            << "at: " << index << "\n";
      }

    for (unsigned i = 0; i < 42; i++) {
      unsigned index = i;
      EXPECT_NEAR(B_grad.getData()[index],
                  b_42_1_grad_matrixsubtraction_graphgradient_test_7[index],
                  1e-5)
          << "at: " << index << "\n";
    }

    for (unsigned j = 0; j < 32; j++)
      for (unsigned i = 0; i < 42; i++) {
        unsigned index = i + j * 42;
        EXPECT_NEAR(C_grad.getData()[index],
                    c_42_32_grad_matrixsubtraction_graphgradient_test_7[index],
                    1e-6)
            << "at: " << index << "\n";
      }
    for (unsigned j = 0; j < 32; j++)
      for (unsigned i = 0; i < 42; i++) {
        unsigned index = i + j * 42;
        EXPECT_NEAR(E_grad.getData()[index],
                    e_42_32_grad_matrixsubtraction_graphgradient_test_7[index],
                    1e-6)
            << "at: " << index << "\n";
      }
    for (unsigned j = 0; j < 32; j++)
      for (unsigned i = 0; i < 42; i++) {
        unsigned index = i + j * 42;
        EXPECT_NEAR(F_grad.getData()[index],
                    f_42_32_grad_matrixsubtraction_graphgradient_test_7[index],
                    1e-6)
            << "at: " << index << "\n";
      }
  }
}
