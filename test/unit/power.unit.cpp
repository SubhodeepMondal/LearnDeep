#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Eager_MatrixPower_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_power[] = {0.1402803,  0.90385769, 0.53581513, 0.35839198,
                              0.02434182, 0.02433429, 0.00337371, 0.75026112,
                              0.36133926, 0.50136678, 0.00042372, 0.94072512,
                              0.69296075, 0.0450879,  0.03306032, 0.03363721};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  A.tensor_of(a);

  C = A.pow(2);

  auto *tensorC_power = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_power->getData()[i], c_power[i], 0.0001);
  }
}

TEST_F(MathTest, Graph_MatrixPower_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t c_power[] = {0.1402803,  0.90385769, 0.53581513, 0.35839198,
                              0.02434182, 0.02433429, 0.00337371, 0.75026112,
                              0.36133926, 0.50136678, 0.00042372, 0.94072512,
                              0.69296075, 0.0450879,  0.03306032, 0.03363721};

  tf::tensor A, C;
  A.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  {
    tf::graph_context ctx;

    C = A.pow(2);

    ctx.run();

    auto *tensorC_power = static_cast<Tensor<std::float64_t> *>(C.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorC_power->getData()[i], c_power[i], 0.0001);
    }
  }
}

TEST_F(MathTest, Eager_MatrixPower_Grad_2D) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985,
                        0.83244264, 0.21233911, 0.18182497, 0.18340451};

  std::float64_t b_power[] = {0.1402803,  0.90385769, 0.53581513, 0.35839198,
                              0.02434182, 0.02433429, 0.00337371, 0.75026112,
                              0.36133926, 0.50136678, 0.00042372, 0.94072512,
                              0.69296075, 0.0450879,  0.03306032, 0.03363721};

  std::float64_t c_power[] = {0.00276051, 0.73841443, 0.15383137, 0.04603359,
                              0.00001442, 0.00001441, 0.00000004, 0.42231579,
                              0.04717864, 0.12602789, 0.00000000, 0.83250763,
                              0.33275601, 0.00009166, 0.00003613, 0.00003806};

  std::float64_t a_grad[] = {0.04422247, 4.66016610, 1.26092334, 0.46136746,
                             0.00055467, 0.00055424, 0.00000397, 2.92538039,
                             0.47091131, 1.06792346, 0.00000002, 5.15001038,
                             2.39840677, 0.00259001, 0.00119239, 0.00124509};

  std::float64_t b_grad[] = {0.05903569, 2.45087618, 0.86129356, 0.38533444,
                             0.00177757, 0.00177647, 0.00003415, 1.68867522,
                             0.39169818, 0.75410593, 0.00000054, 2.65489126,
                             1.44058380, 0.00609876, 0.00327895, 0.00339439};

  tf::tensor A, A_grad, B, B_grad, C, C_grad;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.gradient_required(true);
  C.gradient_required(true);

  A.tensor_of(a);
  B.tensor_of(c_power);
  {
    tf::graph_context ctx;

    B = A.pow(2);
    C = B.pow(3);

    ctx.run();

    auto *tensorB_power = static_cast<Tensor<std::float64_t> *>(B.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorB_power->getData()[i], b_power[i], 0.0001);
    }

    auto *tensorC_power = static_cast<Tensor<std::float64_t> *>(C.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorC_power->getData()[i], c_power[i], 0.0001);
    }
    ctx.initialize_gradient();
    ctx.compute_gradient();

    A_grad = ctx.get_gradient(A);
    B_grad = ctx.get_gradient(B);

    auto *tensorA_grad = static_cast<Tensor<std::float64_t> *>(A_grad.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorA_grad->getData()[i], a_grad[i], 0.0001);
    }
    auto *tensorB_grad = static_cast<Tensor<std::float64_t> *>(B_grad.ptr);
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(tensorB_grad->getData()[i], b_grad[i], 0.0001);
    }
  }
}