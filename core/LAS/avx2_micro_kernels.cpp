#include "avx2_micro_kernels.h"
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
  std::cout << "avx256 kernel for scale is running....\n";
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