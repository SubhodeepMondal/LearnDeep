#ifndef AVX2_MICRO_KERNEL
#define AVX2_MICRO_KERNEL

#include <stdfloat>

namespace avx2 {
void avx2_matmul_conventional_f64(std::float64_t **, unsigned *);

void avx2_matmul_f64(std::float64_t **, unsigned *);

void avx2_add_f64(std::float64_t **, unsigned *);

void avx2_add_broadcast_f64(std::float64_t **ptr, const unsigned nDimA,
                            const unsigned *dimA, const unsigned nDimB,
                            const unsigned *dimB, const bool isBroadCast);

void avx2_sub_f64(std::float64_t **, unsigned *);

void avx2_sub_broadcast_f64(std::float64_t **ptr, const unsigned nDimA,
                            const unsigned *dimA, const unsigned nDimB,
                            const unsigned *dimB, const bool isBroadCast);

void avx2_mul_f64(std::float64_t **, const unsigned *);

void avx2_mul_broadcast_f64(std::float64_t **ptr, const unsigned nDimA,
                            const unsigned *dimA, const unsigned nDimB,
                            const unsigned *dimB, const bool isBroadCast);

void avx2_div_broadcast_f64(std::float64_t **ptr, const unsigned nDimA,
                            const unsigned *dimA, const unsigned nDimB,
                            const unsigned *dimB, const bool isBroadCast);

void avx2_scale_f64(std::float64_t **, unsigned *);

void avx2_sqrt_f64(std::float64_t **, unsigned *);

/**
 * @brief does reduction sum on a given axis with avx optimization
 * @param ptr pointer to pointer of long float pointing to
 *            ptr[0] input
 *            ptr[1] output
 * @param arr enoded integers
 *            arr[0]: n no of dimensions
 *            arr[1-n]: value of each dimensions
 *            arr[n]: reduction axis
 * @return void this funciton returns nothing
 */
void avx2_reduce_sum_f64(std::float64_t *const *const, unsigned const *const);

void avx2_relu_f64(std::float64_t **, const unsigned nDim, unsigned const *);

void avx2_sigmoid_f64(std::float64_t **, unsigned *);

void avx2_softmax_f64(std::float64_t *const *const, unsigned const axis,
                      unsigned const vector_length, unsigned const inner_stride,
                      unsigned const outer_stride);

void avx2_greater_than_zero_f64(std::float64_t *const *const ptr,
                                unsigned *const dims, unsigned const nDims);

} // namespace avx2

#endif //