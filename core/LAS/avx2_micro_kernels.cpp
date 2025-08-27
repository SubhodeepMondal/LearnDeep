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
  std::cout << "avx256 kernel for conventional matmul is running....\n";
// std::cout << omp_get_max_threads() << "\n";
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
  std::cout << "avx256 kernel for matmul is running....\n";
  // std::cout << omp_get_max_threads() << "\n";
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
  std::cout << "avx256 kernel for add is running....\n";
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
  std::cout << "avx256 kernel for sub is running....\n";

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

  std::cout << "avx256 kernel for mul is running....\n";

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

  std::cout << "avx256 kernel for scale is running....\n";

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

  std::cout << "avx256 kernel for sqrt is running....\n";
#pragma omp parallel for
  for (i = 0; i <= n_elements - 4; i += 4) {
    __m256d c_arr =
        _mm256_sqrt_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(a + i)));
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

  std::cout << "avx256 kernel for relu is running....\n";
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

__m256d exp256_pd(__m256d x) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d c1 = _mm256_set1_pd(1.0);
    const __m256d c2 = _mm256_set1_pd(0.5);
    const __m256d c3 = _mm256_set1_pd(1.0/6.0);
    const __m256d c4 = _mm256_set1_pd(1.0/24.0);

    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x3 = _mm256_mul_pd(x2, x);
    __m256d x4 = _mm256_mul_pd(x2, x2);

    __m256d res = _mm256_add_pd(one,
                  _mm256_add_pd(_mm256_mul_pd(c1, x),
                  _mm256_add_pd(_mm256_mul_pd(c2, x2),
                  _mm256_add_pd(_mm256_mul_pd(c3, x3),
                                _mm256_mul_pd(c4, x4)))));
    return res;
}

void avx2::avx2_sigmoid_f64(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *a, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  c = ptr[1];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  omp_set_num_threads(std::thread::hardware_concurrency());

  std::cout << "avx256 kernel for sigmoid is running....\n";
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

  std::cout << "avx256 kernel for softmax is running....\n";
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
                            std::exp(a[i -  2]) + std::exp(a[i - 3]));
}