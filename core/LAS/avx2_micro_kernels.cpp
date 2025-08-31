#ifdef CUDA_ENABLED
#include <LAS/gpu_interface.cuh>
#endif

#include "avx2_micro_kernels.h"
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <thread>

#include <absl/log/log.h>

void avx2::avx2_matmul_conventional_f64(std::float64_t **ptr, unsigned *arr) {
  // x output row size
  // y k row
  // z output column size
  // omp_set_num_threads(12);
  std::float64_t *a, *b, *c;
  unsigned x, y, z;
  double temp[4];
  double sum;

  a = ptr[0];
  b = ptr[1];
  c = ptr[2];

  x = arr[0];
  y = arr[1];
  z = arr[2];
  LOG(INFO) << "avx256 kernel for conventional matmul is running....\n";
// LOG(INFO) << omp_get_max_threads() << "\n";
#pragma omp parallel for
  for (int j = 0; j < z; j++) {
    for (int i = 0; i < x; i++) {
      for (int k = 0; k <= y - 4; k += 4) {

        __m128i indices = _mm_setr_epi32(i + k * x, i + (k + 1) * x,
                                         i + (k + 2) * x, i + (k + 3) * x);

        __m256d mul_temp = _mm256_mul_pd(
            _mm256_loadu_pd(reinterpret_cast<const double *>(a + (k + j * y))),
            _mm256_i32gather_pd(reinterpret_cast<const double *>(b), indices,
                                8));
        _mm256_storeu_pd(temp, mul_temp);
        for (int l = 1; l < 4; l++)
          temp[0] += temp[l];
      }
      // _mm256_storeu_pd(reinterpret_cast<double *>(c + (i + j * x)),
      // sum_temp);
      sum = 0;
      for (int k = y - (y % 4); k < y; k++)
        sum += a[k + j * y] * b[i + k * x];

      sum += temp[0];
      c[i + j * x] = sum;
    }
  }
}

void avx2::avx2_matmul_f64(std::float64_t **ptr, unsigned *arr) {
  // x output row size
  // y k row
  // z output column size
  // omp_set_num_threads(12);
  std::float64_t *a, *b, *c;
  a = ptr[0];
  b = ptr[1];
  c = ptr[2];
  unsigned x, y, z;
  x = arr[0];
  y = arr[1];
  z = arr[2];
  LOG(INFO) << "avx256 kernel for matmul is running....\n";
  // LOG(INFO) << omp_get_max_threads() << "\n";
  memset(c, 0, sizeof(std::float64_t) * x * z);
#pragma omp parallel proc_bind(close)
  {
#pragma omp for
    for (int k = 0; k < z; k++) {
      for (int j = 0; j < y; j++) {
        for (int i = 0; i <= x - 4; i += 4) {
          __m256d c_arr = _mm256_mul_pd(
              _mm256_loadu_pd(
                  reinterpret_cast<const double *>(b + (i + j * x))),
              _mm256_set1_pd(static_cast<const double>(a[j + k * y])));

          __m256d temp =
              _mm256_add_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(
                                c + (i + k * x))),
                            c_arr);
          _mm256_storeu_pd(reinterpret_cast<double *>(c + (i + k * x)), temp);
        }
        for (int i = x - (x % 4); i < x; i++)
          c[i + k * x] += a[j + k * y] * b[i + j * x];
      }
    }
  }
}

void avx2::avx2_add_f64(std::float64_t **ptr, unsigned *arr) {

  std::float64_t *a, *b, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  b = ptr[1];
  c = ptr[2];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  omp_set_num_threads(std::thread::hardware_concurrency());
  LOG(INFO) << "avx256 kernel for add is running....\n";
#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d c_arr =
        _mm256_add_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(a + i)),
                      _mm256_loadu_pd(reinterpret_cast<const double *>(b + i)));

    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }

  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = a[i] + b[i];
}

void avx2::avx2_sub_f64(std::float64_t **ptr, unsigned *arr) {

  std::float64_t *a, *b, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  b = ptr[1];
  c = ptr[2];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  omp_set_num_threads(std::thread::hardware_concurrency());
  LOG(INFO) << "avx256 kernel for sub is running....\n";

#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d c_arr =
        _mm256_sub_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(a + i)),
                      _mm256_loadu_pd(reinterpret_cast<const double *>(b + i)));

    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }

  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = a[i] + b[i];
}

void avx2::avx2_mul_f64(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *a, *b, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  b = ptr[1];
  c = ptr[2];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for mul is running....\n";

#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d c_arr =
        _mm256_mul_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(a + i)),
                      _mm256_loadu_pd(reinterpret_cast<const double *>(b + i)));

    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }

  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = a[i] + b[i];
}

void avx2::avx2_scale_f64(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *a, b, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  b = ptr[1][0];
  c = ptr[2];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for scale is running....\n";

#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d c_arr =
        _mm256_mul_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(a + i)),
                      _mm256_set1_pd(static_cast<const double>(b)));

    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }

  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = a[i] * b;
}

void avx2::avx2_sqrt_f64(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *a, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  c = ptr[1];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for sqrt is running....\n";
#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d c_arr = _mm256_sqrt_pd(
        _mm256_loadu_pd(reinterpret_cast<const double *>(a + i)));
    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }
  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = std::sqrt(a[i]);
}

void avx2::avx2_relu_f64(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *a, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  c = ptr[1];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for relu is running....\n";
#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d zero = _mm256_setzero_pd();
    __m256d c_arr = _mm256_max_pd(
        _mm256_loadu_pd(reinterpret_cast<const double *>(a + i)), zero);
    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }
  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = std::fmax(a[i], 0.0);
}

// #include <immintrin.h>

// scale "a" by 2^k, where k is a double vector (integer values stored as
// double)
static inline __m256d mul_pow2_pd(__m256d a, __m256d k_real) {
    alignas(32) double tmp[4];
    _mm256_storeu_pd(tmp, k_real);   // dump vector to array

    int64_t i0 = (int64_t)tmp[0];
    int64_t i1 = (int64_t)tmp[1];
    int64_t i2 = (int64_t)tmp[2];
    int64_t i3 = (int64_t)tmp[3];

    uint64_t e0 = (uint64_t)(i0 + 1023) << 52;
    uint64_t e1 = (uint64_t)(i1 + 1023) << 52;
    uint64_t e2 = (uint64_t)(i2 + 1023) << 52;
    uint64_t e3 = (uint64_t)(i3 + 1023) << 52;

    alignas(32) uint64_t powbits[4] = { e0, e1, e2, e3 };
    __m256d pow2k = _mm256_castsi256_pd(_mm256_load_si256((__m256i*)powbits));

    return _mm256_mul_pd(a, pow2k);
}

// vectorized exp for 4 doubles using AVX2
inline __m256d exp256_pd(__m256d x) {
  const __m256d ln2 = _mm256_set1_pd(0.6931471805599453);
  const __m256d inv_ln2 = _mm256_set1_pd(1.4426950408889634); // 1/ln(2)

  // Range reduction: k = round(x / ln2)
  __m256d k_real = _mm256_round_pd(
      _mm256_mul_pd(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  // r = x - k * ln2
  __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(k_real, ln2));

  // Polynomial approximation for exp(r), r in [-0.35,0.35]
  const __m256d c1 = _mm256_set1_pd(1.0);
  const __m256d c2 = _mm256_set1_pd(1.0 / 2.0);
  const __m256d c3 = _mm256_set1_pd(1.0 / 6.0);
  const __m256d c4 = _mm256_set1_pd(1.0 / 24.0);
  const __m256d c5 = _mm256_set1_pd(1.0 / 120.0);
  const __m256d c6 = _mm256_set1_pd(1.0 / 720.0);
  const __m256d c7 = _mm256_set1_pd(1.0 / 5040.0);

  __m256d r2 = _mm256_mul_pd(r, r);
  __m256d r3 = _mm256_mul_pd(r2, r);
  __m256d r4 = _mm256_mul_pd(r2, r2);
  __m256d r5 = _mm256_mul_pd(r4, r);
  __m256d r6 = _mm256_mul_pd(r3, r3);
  __m256d r7 = _mm256_mul_pd(r6, r);

  __m256d poly = _mm256_add_pd(
      c1,
      _mm256_add_pd(
          r,
          _mm256_add_pd(
              _mm256_mul_pd(c2, r2),
              _mm256_add_pd(
                  _mm256_mul_pd(c3, r3),
                  _mm256_add_pd(
                      _mm256_mul_pd(c4, r4),
                      _mm256_add_pd(_mm256_mul_pd(c5, r5),
                                    _mm256_add_pd(_mm256_mul_pd(c6, r6),
                                                  _mm256_mul_pd(c7, r7))))))));

  // scale by 2^k
  return mul_pow2_pd(poly, k_real);
}

void avx2::avx2_sigmoid_f64(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *a, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  c = ptr[1];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  // unsigned num_threads = (n_elements / 4) >=
  // std::thread::hardware_concurrency()
  //                            ? std::thread::hardware_concurrency()
  //                            : (n_elements / 4) - 1;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for sigmoid is running....\n";
#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d one = _mm256_set1_pd(1.0);
    __m256d neg = _mm256_set1_pd(-1.0);
    __m256d x = _mm256_loadu_pd(reinterpret_cast<const double *>(a + i));
    __m256d exp_val = exp256_pd(_mm256_mul_pd(neg, x));
    __m256d denom = _mm256_add_pd(one, exp_val);
    __m256d c_arr = _mm256_div_pd(one, denom);
    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }
  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = 1 / (1 + std::exp(-a[i]));
}

void avx2::avx2_softmax_f64(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *a, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  c = ptr[1];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for softmax is running....\n";
#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d one = _mm256_set1_pd(1.0);
    __m256d x = _mm256_loadu_pd(reinterpret_cast<const double *>(a + i));
    __m256d exp_val = exp256_pd(x);
    __m256d denom = _mm256_hadd_pd(exp_val, exp_val);
    denom = _mm256_hadd_pd(denom, denom);
    __m256d c_arr = _mm256_div_pd(exp_val, denom);
    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }
  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = std::exp(a[i]) / (std::exp(a[i]) + std::exp(a[i - 1]) +
                             std::exp(a[i - 2]) + std::exp(a[i - 3]));
}