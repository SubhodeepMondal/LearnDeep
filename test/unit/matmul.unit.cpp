#include <gtest/gtest.h>
#include <tensor.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, MatrixMultiplication_2D) {

  std::float64_t a[] = {0.42602198, 0.51120308, 0.66381781, 0.79000792,
                        0.73980886, 0.1366799,  0.3818528,  0.40564105,
                        0.79132994, 0.1810338,  0.52634304, 0.7717289,
                        0.18137833, 0.35597476, 0.79365669, 0.16725214};

  std::float64_t b[] = {0.09328713, 0.4622637,  0.78946444, 0.14203406,
                        0.84087653, 0.19860089, 0.85865357, 0.1971424,
                        0.00456894, 0.05930002, 0.91838192, 0.94089892,
                        0.47830435, 0.49946382, 0.6011153,  0.97711538};

  std::float64_t c_mul[] = {0.85049821, 0.73240467, 1.85979968, 1.55780379,
                            0.37971011, 0.59437844, 1.29593722, 0.88766646,
                            0.59757409, 0.8184194,  1.72745414, 1.39738902,
                            0.39987468, 0.28514178, 1.27826852, 1.00611498};

  tf::tensor A, B, C;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);

  tf::graph g_matmul;
  g_matmul.tf_create_graph();

  g_matmul.graph_start_recording_session();

  C = A.matmul(B);

  g_matmul.graph_end_recording_session();
  g_matmul.graph_execute();

  auto *tensorC_add = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_add->getData()[i], c_mul[i], 0.0001);
  }
}
