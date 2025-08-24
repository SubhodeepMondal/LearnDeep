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

TEST_F(MathTest, MatrixPower_2D) {

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

  tf::graph g_power;
  g_power.tf_create_graph();

  // g_power.graph_start_recording_session();

  C = A.pow(g_power, 2);

  // g_power.graph_end_recording_session();

  g_power.graph_execute();
  g_power.graph_clear();

  auto *tensorC_power = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_power->getData()[i], c_power[i], 0.0001);
  }
}
