#include "LinearAlgebraFixtures.unit.hpp"
#include <gtest/gtest.h>
#include <tensor.h>

TEST_F(MathTest, Eager_MatrixAddition_2D) {

  std::float64_t a[] = {0.42602198, 0.51120308, 0.66381781, 0.79000792,
                        0.73980886, 0.1366799,  0.3818528,  0.40564105,
                        0.79132994, 0.1810338,  0.52634304, 0.7717289,
                        0.18137833, 0.35597476, 0.79365669, 0.16725214};

  std::float64_t b[] = {0.09328713, 0.4622637,  0.78946444, 0.14203406,
                        0.84087653, 0.19860089, 0.85865357, 0.1971424,
                        0.00456894, 0.05930002, 0.91838192, 0.94089892,
                        0.47830435, 0.49946382, 0.6011153,  0.97711538};

  std::float64_t c_add[] = {0.51930911, 0.97346678, 1.45328225, 0.93204199,
                            1.58068539, 0.33528079, 1.24050637, 0.60278345,
                            0.79589888, 0.24033382, 1.44472496, 1.71262782,
                            0.65968268, 0.85543858, 1.39477199, 1.14436752};

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);

  C = A.add(B);

  auto *tensorC_matmul = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(C.getPtr()[i], c_add[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_MatrixAddition_2D) {

  std::float64_t a[] = {0.42602198, 0.51120308, 0.66381781, 0.79000792,
                        0.73980886, 0.1366799,  0.3818528,  0.40564105,
                        0.79132994, 0.1810338,  0.52634304, 0.7717289,
                        0.18137833, 0.35597476, 0.79365669, 0.16725214};

  std::float64_t b[] = {0.09328713, 0.4622637,  0.78946444, 0.14203406,
                        0.84087653, 0.19860089, 0.85865357, 0.1971424,
                        0.00456894, 0.05930002, 0.91838192, 0.94089892,
                        0.47830435, 0.49946382, 0.6011153,  0.97711538};

  std::float64_t c_add[] = {0.51930911, 0.97346678, 1.45328225, 0.93204199,
                            1.58068539, 0.33528079, 1.24050637, 0.60278345,
                            0.79589888, 0.24033382, 1.44472496, 1.71262782,
                            0.65968268, 0.85543858, 1.39477199, 1.14436752};

  tf::tensor A, B, C, D;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);

  tf::graph g_add;
  g_add.tf_create_graph();

  C = A.add(g_add, B);

  g_add.graph_execute();

  auto *tensorC_add = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_add->getData()[i], c_add[i], 0.0001);
  }
  g_add.graph_clear();
}

TEST_F(MathTest, Graph_MatrixAddition_Grad_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t b[] = {0.30424224, 0.52475643, 0.43194502, 0.29122914,
                        0.61185289, 0.13949386, 0.29214465, 0.36636184,
                        0.45606998, 0.78517596, 0.19967378, 0.51423444,
                        0.59241457, 0.04645041, 0.60754485, 0.17052412};

  std::float64_t e[] = {0.1402803,  0.90385769, 0.53581513, 0.35839198,
                        0.02434182, 0.02433429, 0.00337371, 0.75026112,
                        0.36133926, 0.50136678, 0.00042372, 0.94072512,
                        0.69296075, 0.0450879,  0.03306032, 0.03363721};

  std::float64_t f_res[] = {0.06463352, 1.96771076, 0.72589764, 0.28381060,
                            0.01435258, 0.00212471, 0.00041382, 1.13975909,
                            0.40384725, 1.11794322, 0.00002056, 2.07212043,
                            1.40686144, 0.00301963, 0.02060004, 0.00421358};

  std::float64_t d_a[] = {0.0130863,  0.417821,   0.423007,    0.0509039,
                          0.0204685,  0.00483281, 0.00289685,  0.147908,
                          0.00165094, 0.0297311,  0.000389137, 0.885127,
                          0.331446,   0.0225198,  0.0198731,   0.0328674};

  std::float64_t d_b[] = {0.0597625, 0.462055,   0.355684,    0.283133,
                          0.0180083, 0.00332601, 0.00128826,  0.304337,
                          0.285939,  0.0907643,  0.000223022, 0.725985,
                          0.125688,  0.0160502,  0.0262385,   0.0056259};

  std::float64_t d_c[] = {0.19043959, 2.66723115, 1.24731221, 0.63785718,
                          0.03738278, 0.01438100, 0.00236313, 1.84945065,
                          0.76400488, 1.49733041, 0.00018666, 2.79234364,
                          1.97474024, 0.02333655, 0.05219364, 0.02381035};

  std::float64_t d_d[] = {0.678782, 1.47547,  1.16394,  0.889888,
                          0.767872, 0.295488, 0.350228, 1.23254,
                          1.05718,  1.49325,  0.220258, 1.48414,
                          1.42486,  0.258790, 0.789370, 0.353929};

  std::float64_t d_e[] = {0.46074549, 2.17701390, 1.35475390, 0.79189998,
                          0.58962669, 0.08731338, 0.12265983, 1.51914989,
                          1.11764012, 2.22979120, 0.04851371, 2.20268428,
                          2.03021807, 0.06697202, 0.62310471, 0.12526548};

  tf::tensor A, A_grad, B, B_grad, C, C_grad, D, D_grad, E, E_grad, F, F_grad;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);
  D.tf_create(tf_float64, 4, 4);
  E.tf_create(tf_float64, 4, 4);
  F.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);
  E.tensor_of(e);

  tf::graph g_add;
  g_add.tf_create_graph();

  // A         B
  // |         |
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

  C = A.add(g_add, B);
  D = C.mul(g_add, E);
  F = C.mul(g_add, D);

  g_add.graph_execute();

  auto *tensorF_res = static_cast<Tensor<std::float64_t> *>(F.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorF_res->getData()[i], f_res[i], 0.0001);
  }

  g_add.graph_initialize_gradient();
  g_add.graph_compute_gradient();
  // g_add.graph_traverse_gradient();

  A_grad = g_add.graph_get_gradient(A);
  B_grad = g_add.graph_get_gradient(B);
  C_grad = g_add.graph_get_gradient(C);
  E_grad = g_add.graph_get_gradient(E);
  D_grad = g_add.graph_get_gradient(D);

  // C_grad.print_data();

  auto *tensorA_grad = static_cast<Tensor<std::float64_t> *>(A_grad.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorA_grad->getData()[i], d_c[i], 0.0001);
  }

  auto *tensorB_grad = static_cast<Tensor<std::float64_t> *>(B_grad.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorB_grad->getData()[i], d_c[i], 0.0001);
  }

  auto *tensorC_grad = static_cast<Tensor<std::float64_t> *>(C_grad.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_grad->getData()[i], d_c[i], 0.0001);
  }

  auto *tensorE_grad = static_cast<Tensor<std::float64_t> *>(E_grad.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorE_grad->getData()[i], d_e[i], 0.0001);
  }

  auto *tensorD_grad = static_cast<Tensor<std::float64_t> *>(D_grad.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorD_grad->getData()[i], d_d[i], 0.0001);
  }
  g_add.graph_clear();
}
