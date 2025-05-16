#ifndef AVX2_MICRO_KERNEL
#define AVX2_MICRO_KERNEL

#include <stdfloat>

namespace avx2 {
void avx2_matmul_conventional_f64(std::float64_t **, unsigned *);

void avx2_matmul_f64(std::float64_t **, unsigned *);

void avx2_add_f64(std::float64_t **, unsigned *);

void avx2_sub_f64(std::float64_t **, unsigned *);

void avx2_mul_f64(std::float64_t **, unsigned *);

void avx2_scale_f64(std::float64_t **, unsigned *);

void avx2_relu_f64(std::float64_t **, unsigned *);

void avx2_sigmoid_f64(std::float64_t **, unsigned *);

void avx2_softmax_f64(std::float64_t **, unsigned *);

} // namespace avx2

#endif