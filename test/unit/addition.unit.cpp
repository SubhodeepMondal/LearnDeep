#include "LinearAlgebraFixtures.unit.hpp"
#include <gtest/gtest.h>
#include <tensor.h>

TEST_F(MathTest, MatrixAddition_2D) {

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

  g_add.graph_start_recording_session();

  C = A.add(B);

  g_add.graph_end_recording_session();
  g_add.graph_execute();
  // g_add.graph_clear();

  auto *tensorC_add = static_cast<Tensor<std::float64_t> *>(C.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(tensorC_add->getData()[i], c_add[i], 0.0001);
  }
}
