#include "LinearAlgebraFixtures.unit.hpp"
#include <gtest/gtest.h>
#include <tensor.h>

TEST_F(MathTest, MatrixInitialization_test) {
  std::float64_t a[] = {0.00805433, 0.71604533, 0.53269858, 0.34661127,
                        0.56137041, 0.14270995, 0.62471964, 0.31898735,
                        0.32166846, 0.11043887, 0.43579798, 0.68246876,
                        0.91418964, 0.23475763, 0.45345491, 0.52145146};

  std::float64_t b[] = {0.49704563, 0.07919775, 0.28764169, 0.48468005,
                        0.55964598, 0.51628139, 0.81488106, 0.51503196,
                        0.2614797,  0.78048695, 0.91069071, 0.08636421,
                        0.19244527, 0.8450769,  0.84808829, 0.6912999};

  std::float64_t a_temp[] = {0.00805433, 0.71604533, 0.53269858, 0.34661127,
                             0.56137041, 0.14270995, 0.62471964, 0.31898735,
                             0.32166846, 0.11043887, 0.43579798, 0.68246876,
                             0.91418964, 0.23475763, 0.45345491, 0.52145146};

  std::float64_t b_temp[] = {0.49704563, 0.07919775, 0.28764169, 0.48468005,
                             0.55964598, 0.51628139, 0.81488106, 0.51503196,
                             0.2614797,  0.78048695, 0.91069071, 0.08636421,
                             0.19244527, 0.8450769,  0.84808829, 0.6912999};

  // Check if the dimensions of A and B are as expected
  unsigned a_dims = A.getNoOfDimensions();
  unsigned b_dims = B.getNoOfDimensions();
  ASSERT_EQ(a_dims, 2);
  ASSERT_EQ(b_dims, 2);

  // Check if no of elements in A and B are as expected
  ASSERT_EQ(A.getNoOfElem(), 16);
  ASSERT_EQ(B.getNoOfElem(), 16);

  A.tensor_of(a);
  B.tensor_of(b);

  // Check if the data in A and B matches the expected values
  auto *a_ptr = static_cast<Tensor<std::float64_t> *>(A.ptr);
  auto *b_ptr = static_cast<Tensor<std::float64_t> *>(B.ptr);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(a_ptr->getData()[i], a_temp[i]);
    EXPECT_EQ(b_ptr->getData()[i], b_temp[i]);
  }
}