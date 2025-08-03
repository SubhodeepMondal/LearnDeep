#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, MatrixElementWiseMultiplication_2D) {

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

  tf::graph g_mul;
  g_mul.tf_create_graph();

  g_mul.graph_start_recording_session();

  C = A.mul(B);

  g_mul.graph_end_recording_session();
  g_mul.graph_execute();
  g_mul.graph_clear();

  auto *tensorC_mul = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_mul->getData()[i], c_mul[i], 0.0001);
  }
}
