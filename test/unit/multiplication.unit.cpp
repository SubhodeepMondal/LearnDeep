#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eager_MatrixHarmandMultiplication_2D) {

  std::float64_t a[] = {0.42602198, 0.51120308, 0.66381781, 0.79000792,
                        0.73980886, 0.1366799,  0.3818528,  0.40564105,
                        0.79132994, 0.1810338,  0.52634304, 0.7717289,
                        0.18137833, 0.35597476, 0.79365669, 0.16725214};

  std::float64_t b[] = {0.09328713, 0.4622637,  0.78946444, 0.14203406,
                        0.84087653, 0.19860089, 0.85865357, 0.1971424,
                        0.00456894, 0.05930002, 0.91838192, 0.94089892,
                        0.47830435, 0.49946382, 0.6011153,  0.97711538};

  std::float64_t c_add[] = {0.03974237, 0.23631063, 0.52406055, 0.11220804,
                            0.6220879,  0.02714475, 0.32787927, 0.07996905,
                            0.00361554, 0.01073531, 0.48338393, 0.72611889,
                            0.08675404, 0.17779651, 0.47707918, 0.16342464};

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  // C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);

  C = A * B;

  auto *tensorC_matmul = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_matmul->getData()[i], c_add[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_MatrixElementWiseMultiplication_2D) {

  std::float64_t a[] = {0.42602198, 0.51120308, 0.66381781, 0.79000792,
                        0.73980886, 0.1366799,  0.3818528,  0.40564105,
                        0.79132994, 0.1810338,  0.52634304, 0.7717289,
                        0.18137833, 0.35597476, 0.79365669, 0.16725214};

  std::float64_t b[] = {0.09328713, 0.4622637,  0.78946444, 0.14203406,
                        0.84087653, 0.19860089, 0.85865357, 0.1971424,
                        0.00456894, 0.05930002, 0.91838192, 0.94089892,
                        0.47830435, 0.49946382, 0.6011153,  0.97711538};

  std::float64_t c_mul[] = {0.03974237, 0.23631063, 0.52406055, 0.11220804,
                            0.6220879,  0.02714475, 0.32787927, 0.07996905,
                            0.00361554, 0.01073531, 0.48338393, 0.72611889,
                            0.08675404, 0.17779651, 0.47707918, 0.16342464};

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);
  {
    tf::graph_context ctx;

    C = A.mul(B);

    ctx.run();

    auto *tensorC_mul = static_cast<Tensor<std::float64_t> *>(C.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorC_mul->getData()[i], c_mul[i], 0.0001);
    }
  }
}

TEST_F(MathTest, Graph_MatrixElementWiseMultiplication_Grad_2D) {

  std::float64_t a[] = {0.42602198, 0.51120308, 0.66381781, 0.79000792,
                        0.73980886, 0.1366799,  0.3818528,  0.40564105,
                        0.79132994, 0.1810338,  0.52634304, 0.7717289,
                        0.18137833, 0.35597476, 0.79365669, 0.16725214};

  std::float64_t b[] = {0.09328713, 0.4622637,  0.78946444, 0.14203406,
                        0.84087653, 0.19860089, 0.85865357, 0.1971424,
                        0.00456894, 0.05930002, 0.91838192, 0.94089892,
                        0.47830435, 0.49946382, 0.6011153,  0.97711538};

  std::float64_t e[] = {0.1402803,  0.90385769, 0.53581513, 0.35839198,
                        0.02434182, 0.02433429, 0.00337371, 0.75026112,
                        0.36133926, 0.50136678, 0.00042372, 0.94072512,
                        0.69296075, 0.0450879,  0.03306032, 0.03363721};

  std::float64_t d_a[] = {0.0130863,  0.417821,   0.423007,    0.0509039,
                          0.0204685,  0.00483281, 0.00289685,  0.147908,
                          0.00165094, 0.0297311,  0.000389137, 0.885127,
                          0.331446,   0.0225198,  0.0198731,   0.0328674};

  std::float64_t d_b[] = {0.0597625, 0.462055,   0.355684,    0.283133,
                          0.0180083, 0.00332601, 0.00128826,  0.304337,
                          0.285939,  0.0907643,  0.000223022, 0.725985,
                          0.125688,  0.0160502,  0.0262385,   0.00562590};

  std::float64_t d_c[] = {0.1402803,  0.90385769, 0.53581513, 0.35839198,
                          0.02434182, 0.02433429, 0.00337371, 0.75026112,
                          0.36133926, 0.50136678, 0.00042372, 0.94072512,
                          0.69296075, 0.0450879,  0.03306032, 0.03363721};

  std::float64_t d_e[] = {0.03974237, 0.23631063, 0.52406055, 0.11220804,
                          0.6220879,  0.02714475, 0.32787927, 0.07996905,
                          0.00361554, 0.01073531, 0.48338393, 0.72611889,
                          0.08675404, 0.17779651, 0.47707918, 0.16342464};

  tf::tensor A, A_grad, B, B_grad, C, C_grad, D, D_grad, E, E_grad;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);
  D.tf_create(tf_float64, 4, 4);
  E.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);
  E.tensor_of(e);
  {
    tf::graph_context ctx;

    C = A.mul(B);
    D = C.mul(E);

    ctx.run();

    ctx.initialize_gradient();
    ctx.compute_gradient();

    A_grad = ctx.get_gradient(A);
    B_grad = ctx.get_gradient(B);
    C_grad = ctx.get_gradient(C);
    E_grad = ctx.get_gradient(E);

    auto *tensorA_grad = static_cast<Tensor<std::float64_t> *>(A_grad.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorA_grad->getData()[i], d_a[i], 0.0001);
    }

    auto *tensorB_grad = static_cast<Tensor<std::float64_t> *>(B_grad.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorB_grad->getData()[i], d_b[i], 0.0001);
    }

    auto *tensorC_grad = static_cast<Tensor<std::float64_t> *>(C_grad.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorC_grad->getData()[i], d_c[i], 0.0001);
    }

    auto *tensorE_grad = static_cast<Tensor<std::float64_t> *>(E_grad.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorE_grad->getData()[i], d_e[i], 0.0001);
    }
  }
}
