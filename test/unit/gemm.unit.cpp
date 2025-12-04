#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eraph_General_Matrix_Multiplication_2D) {
  // gemm is a general matrix multiplication operation
  // It performs matrix multiplication followed by addition with another matrix
  // A * B + C

  std::float64_t a[] = {0.54671028, 0.18485446, 0.96958463, 0.77513282,
                        0.93949894, 0.89482735, 0.59789998, 0.92187424,
                        0.0884925,  0.19598286, 0.04522729, 0.32533033,
                        0.38867729, 0.27134903, 0.82873751, 0.35675333};

  std::float64_t b[] = {0.28093451, 0.54269608, 0.14092422, 0.80219698,
                        0.07455064, 0.98688694, 0.77224477, 0.19871568,
                        0.00552212, 0.81546143, 0.70685734, 0.72900717,
                        0.77127035, 0.07404465, 0.35846573, 0.11586906};

  std::float64_t c[] = {0.86310343, 0.62329813, 0.33089802, 0.06355835,
                        0.31098232, 0.32518332, 0.72960618, 0.63755747,
                        0.88721274, 0.47221493, 0.11959425, 0.71324479,
                        0.76078505, 0.5612772,  0.77096718, 0.4937956};

  std::float64_t e_gemm[] = {1.63366535, 1.95047941, 1.5139122,  1.33550922,
                             1.35594589, 2.27396337, 2.30612039, 2.11172698,
                             1.17785138, 0.77462247, 0.43200074, 0.89384481,
                             1.1699368,  1.7422208,  1.74897213, 1.50500491};

  tf::tensor A, B, C, D, E;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);
  // D.tf_create(tf_float64, 4, 4);
  // E.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);
  C.tensor_of(c);

  D = A.matmul(B);
  E = D.add(C);

  auto *tensorE_gemm = static_cast<Tensor<std::float64_t> *>(E.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorE_gemm->getData()[i], e_gemm[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_General_Matrix_Multiplication_2D) {
  // gemm is a general matrix multiplication operation
  // It performs matrix multiplication followed by addition with another matrix
  // A * B + C

  std::float64_t a[] = {0.54671028, 0.18485446, 0.96958463, 0.77513282,
                        0.93949894, 0.89482735, 0.59789998, 0.92187424,
                        0.0884925,  0.19598286, 0.04522729, 0.32533033,
                        0.38867729, 0.27134903, 0.82873751, 0.35675333};

  std::float64_t b[] = {0.28093451, 0.54269608, 0.14092422, 0.80219698,
                        0.07455064, 0.98688694, 0.77224477, 0.19871568,
                        0.00552212, 0.81546143, 0.70685734, 0.72900717,
                        0.77127035, 0.07404465, 0.35846573, 0.11586906};

  std::float64_t c[] = {0.86310343, 0.62329813, 0.33089802, 0.06355835,
                        0.31098232, 0.32518332, 0.72960618, 0.63755747,
                        0.88721274, 0.47221493, 0.11959425, 0.71324479,
                        0.76078505, 0.5612772,  0.77096718, 0.4937956};

  std::float64_t e_gemm[] = {1.63366535, 1.95047941, 1.5139122,  1.33550922,
                             1.35594589, 2.27396337, 2.30612039, 2.11172698,
                             1.17785138, 0.77462247, 0.43200074, 0.89384481,
                             1.1699368,  1.7422208,  1.74897213, 1.50500491};

  tf::tensor A, B, C, D, E;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);
  D.tf_create(tf_float64, 4, 4);
  E.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);
  C.tensor_of(c);

  {
    tf::graph_context ctx;
    D = A.matmul(B);
    E = D.add(C);

    ctx.run();

    auto *tensorE_gemm = static_cast<Tensor<std::float64_t> *>(E.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorE_gemm->getData()[i], e_gemm[i], 0.0001);
    }
  }
}
