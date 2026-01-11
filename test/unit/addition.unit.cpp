#include "LinearAlgebraFixtures.unit.hpp"
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "addition.data.hpp"

TEST_F(MathTest, Eager_MatrixAddition1_2D) {

  //--------------- Test 1 -----------------
  /*
   * Tensor A dim [16, 8]
   * Tensor B dim [16, 8]
   * Output   dim [16, 8]
   */
  tf::tensor A_16_8, B_16_8, C_16_8;
  A_16_8.tf_create(tf_float64, 16, 8);
  B_16_8.tf_create(tf_float64, 16, 8);
  C_16_8.tf_create(tf_float64, 16, 8);

  A_16_8.tensor_of(a_double_16_8);
  B_16_8.tensor_of(b_double_16_8);

  C_16_8 = A_16_8.add(B_16_8);

  for (int j = 0; j < 8; j++) {
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(C_16_8.getData()[i + j * 16], out_double_16_8[i + j * 16],
                  0.0001)
          << "at: " << i + j * 16 << " .";
    }
  }
  //----------- End Of Test 1 --------------
}

TEST_F(MathTest, Eager_MatrixAddition2_2D) {

  //--------------- Test 2 -----------------
  /*
   * Tensor A dim [19, 31]
   * Tensor B dim [19, 31]
   * Output   dim [19, 31]
   */
  tf::tensor A_19_31, B_19_31, C_19_31;
  A_19_31.tf_create(tf_float64, 19, 31);
  B_19_31.tf_create(tf_float64, 19, 31);
  C_19_31.tf_create(tf_float64, 19, 31);

  A_19_31.tensor_of(a_double_19_31);
  B_19_31.tensor_of(b_double_19_31);

  C_19_31 = A_19_31.add(B_19_31);

  for (int j = 0; j < 31; j++) {
    for (int i = 0; i < 19; i++) {
      EXPECT_NEAR(C_19_31.getData()[i + j * 19], out_double_19_31[i + j * 19],
                  0.0001)
          << "at" << i + j * 19 << " .";
    }
  }
  //----------- End Of Test 2 --------------
}

TEST_F(MathTest, Eager_MatrixAddition3_3D) {

  //--------------- Test 3 -----------------
  /*
   * Tensor A dim [12, 81, 5]
   * Tensor B dim [12, 81, 5]
   * Output   dim [12, 81, 5]
   */
  tf::tensor A_12_81_5, B_12_81_5, C_12_81_5;
  A_12_81_5.tf_create(tf_float64, 12, 81, 5);
  B_12_81_5.tf_create(tf_float64, 12, 81, 5);
  C_12_81_5.tf_create(tf_float64, 12, 81, 5);

  A_12_81_5.tensor_of(a_double_12_81_5);
  B_12_81_5.tensor_of(b_double_12_81_5);

  C_12_81_5 = A_12_81_5.add(B_12_81_5);

  for (unsigned k = 0; k < 5; k++) {
    for (unsigned j = 0; j < 81; j++) {
      for (unsigned i = 0; i < 12; i++) {
        EXPECT_NEAR(C_12_81_5.getData()[i + j * 12 + k * 12 * 81],
                    out_double_12_81_5[i + j * 12 + k * 12 * 81], 1e-6)
            << "at " << i + j * 12 + k * 12 * 81 << " .";
      }
    }
  }
  //----------- End Of Test 3 --------------
}

TEST_F(MathTest, Eager_MatrixAddition4_3D) {

  //--------------- Test 4 -----------------
  /*
   * Tensor A dim [18, 27, 5]
   * Tensor B dim [18, 27]
   * Output   dim [18, 27, 5]
   */

  tf::tensor A_13_6_5, B_13_6, C_13_6_5;
  A_13_6_5.tf_create(tf_float64, 13, 6, 5);
  B_13_6.tf_create(tf_float64, 13, 6);
  C_13_6_5.tf_create(tf_float64, 13, 6, 5);

  A_13_6_5.tensor_of(a_double_13_6_5);
  B_13_6.tensor_of(b_double_13_6);

  C_13_6_5 = A_13_6_5.add(B_13_6);

  for (unsigned k = 0; k < 5; k++) {
    for (unsigned j = 0; j < 6; j++) {
      for (unsigned i = 0; i < 13; i++) {
        unsigned index = i + j * 13 + k * 13 * 6;
        EXPECT_NEAR(C_13_6_5.getData()[index], out_double_13_6_5[index], 1e-6)
            << "at " << index << " .";
      }
    }
  }
  //----------- End Of Test 3 --------------
}

TEST_F(MathTest, Eager_MatrixAddition5_3D) {

  //--------------- Test 5 -----------------
  /*
   * Tensor A dim [18, 27, 5]
   * Tensor B dim [18, 27]
   * Output   dim [18, 27, 5]
   */
  tf::tensor A_5_3_4, B_5_1_4, C_5_3_4;
  A_5_3_4.tf_create(tf_float64, 5, 3, 4);
  B_5_1_4.tf_create(tf_float64, 5, 1, 4);
  C_5_3_4.tf_create(tf_float64, 5, 3, 4);

  A_5_3_4.tensor_of(a_double_5_3_4);
  B_5_1_4.tensor_of(b_double_5_1_4);

  C_5_3_4 = A_5_3_4.add(B_5_1_4);

  for (unsigned k = 0; k < 4; k++) {
    for (unsigned j = 0; j < 3; j++) {
      for (unsigned i = 0; i < 5; i++) {
        unsigned index = i + j * 5 + k * 5 * 3;
        EXPECT_NEAR(C_5_3_4.getData()[index], out_double_5_3_4[index], 1e-6)
            << "at " << index << " .";
      }
    }
  }
  //----------- End Of Test 5 --------------
}

TEST_F(MathTest, Eager_MatrixAddition6_3D) {

  //--------------- Test 5 -----------------
  /*
   * Tensor A dim [18, 27, 5]
   * Tensor B dim [18, 27]
   * Output   dim [18, 27, 5]
   */
  tf::tensor A_12_3_4_7_9, B_1_3_1_1_9, C_12_3_4_7_9;
  A_12_3_4_7_9.tf_create(tf_float64, 12, 3, 4, 7, 9);
  B_1_3_1_1_9.tf_create(tf_float64, 1, 3, 1, 1, 9);
  C_12_3_4_7_9.tf_create(tf_float64, 12, 3, 4, 7, 9);

  A_12_3_4_7_9.tensor_of(a_double_12_3_4_7_9);
  B_1_3_1_1_9.tensor_of(b_double_1_3_1_1_9);

  C_12_3_4_7_9 = A_12_3_4_7_9.add(B_1_3_1_1_9);

  for (unsigned k = 0; k < 189; k++) {
    for (unsigned j = 0; j < 3; j++) {
      for (unsigned i = 0; i < 12; i++) {
        unsigned index = i + j * 12 + k * 12 * 3;
        EXPECT_NEAR(C_12_3_4_7_9.getData()[index], output_12_3_4_7_9[index],
                    1e-6)
            << "at " << index << " .";
      }
    }
  }
  //----------- End Of Test 5 --------------
}

TEST_F(MathTest, Graph_MatrixAddition1_2D) {

  //--------------- Test 1 -----------------
  tf::tensor A_16_8, B_16_8, C_16_8;
  A_16_8.tf_create(tf_float64, 16, 8);
  B_16_8.tf_create(tf_float64, 16, 8);
  C_16_8.tf_create(tf_float64, 16, 8);

  A_16_8.tensor_of(graph_a_double_16_8);
  B_16_8.tensor_of(graph_b_double_16_8);

  {
    tf::graph_context ctx;

    C_16_8 = A_16_8.add(B_16_8);

    ctx.run();

    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 16; i++) {
        EXPECT_NEAR(C_16_8.getData()[i + j * 16],
                    graph_out_double_16_8[i + j * 16], 0.0001)
            << "at: " << i + j * 19 << " .";
      }
    }
  }
}

TEST_F(MathTest, Graph_MatrixAddition2_2D) {
  //--------------- Test 1 -----------------
  tf::tensor A_56_64, B_56, C_56_64;
  A_56_64.tf_create(tf_float64, 56, 64);
  B_56.tf_create(tf_float64, 56);
  C_56_64.tf_create(tf_float64, 56, 64);

  A_56_64.tensor_of(graph_a_double_56_64);
  B_56.tensor_of(graph_b_double_56);

  {
    tf::graph_context ctx;

    C_56_64 = A_56_64.add(B_56);

    ctx.run();

    for (int j = 0; j < 64; j++) {
      for (int i = 0; i < 56; i++) {
        unsigned index = i + j * 56;
        EXPECT_NEAR(C_56_64.getData()[index], graph_out_double_56_64[index],
                    1e-6)
            << "at: " << index << " .";
      }
    }
  }
}

TEST_F(MathTest, Graph_MatrixAddition_Grad_2D) {

  //--------------- Test 1 -----------------
  tf::tensor A, A_grad, B, B_grad, C, C_grad, D, D_grad, E, E_grad, F, F_grad;
  A.tf_create(tf_float64, 28, 56);
  B.tf_create(tf_float64, 28, 56);
  C.tf_create(tf_float64, 28, 56);
  D.tf_create(tf_float64, 28, 56);
  E.tf_create(tf_float64, 28, 56);
  F.tf_create(tf_float64, 28, 56);

  A.tensor_of(a_double_28_56);
  B.tensor_of(b_double_28_56);
  E.tensor_of(e_double_28_56);

  {

    // Directed acyclic graph for
    // A         B
    //  \       /
    //   \     /
    //      +
    //      |
    //  E   C
    //   \ / \
  //    *  |
    //    |  |
    //    D  C
    //    \ /
    //     *
    //     |
    //     F
    tf::graph_context ctx;
    C = A.add(B);
    D = C.mul(E);
    F = C.mul(D);

    ctx.run();

    for (int j = 0; j < 56; j++)
      for (int i = 0; i < 28; i++)
        EXPECT_NEAR(F.getData()[i + j * 28], f_out_28_56[i + j * 28], 1e-6);

    ctx.initialize_gradient();
    ctx.compute_gradient();

    A_grad = ctx.get_gradient(A);
    B_grad = ctx.get_gradient(B);
    C_grad = ctx.get_gradient(C);
    E_grad = ctx.get_gradient(E);
    D_grad = ctx.get_gradient(D);

    for (int j = 0; j < 56; j++)
      for (int i = 0; i < 28; i++)
        EXPECT_NEAR(A_grad.getData()[i + j * 28], d_a[i + j * 28], 1e-6);

    for (int j = 0; j < 56; j++)
      for (int i = 0; i < 28; i++)
        EXPECT_NEAR(B_grad.getData()[i + j * 28], d_b[i + j * 28], 1e-6);

    for (int j = 0; j < 56; j++)
      for (int i = 0; i < 28; i++)
        EXPECT_NEAR(C_grad.getData()[i + j * 28], d_c[i + j * 28], 1e-6);

    for (int j = 0; j < 56; j++)
      for (int i = 0; i < 28; i++)
        EXPECT_NEAR(E_grad.getData()[i + j * 28], d_e[i + j * 28], 1e-6);

    for (int j = 0; j < 56; j++)
      for (int i = 0; i < 28; i++)
        EXPECT_NEAR(D_grad.getData()[i + j * 28], d_d[i + j * 28], 1e-6);
  }

  //----------- End Of Test 1 --------------
}
