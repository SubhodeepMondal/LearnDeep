#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
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
      for (int k = 0; k + 4 <= y; k += 4) {

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
        for (int i = 0; i + 4 <= x; i += 4) {
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
  unsigned vec_end = (n_elements / 4) * 4;
  omp_set_num_threads(std::thread::hardware_concurrency());
  LOG(INFO) << "avx256 kernel for add is running....\n";
#pragma omp parallel for
  for (i = 0; i < vec_end; i += 4) {
    __m256d c_arr =
        _mm256_add_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(a + i)),
                      _mm256_loadu_pd(reinterpret_cast<const double *>(b + i)));

    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }

  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = a[i] + b[i];
}

void avx2::avx2_add_broadcast_f64(std::float64_t **ptr, const unsigned nDimA,
                                  const unsigned *dimA, const unsigned nDimB,
                                  const unsigned *dimB,
                                  const bool isBroadCast) {
  unsigned grid_x, grid_y;
  unsigned n_dim_A = nDimA;
  unsigned total_lines = 1;
  unsigned total_lines_b = 1;
  std::float64_t *a, *b, *c;
  a = ptr[0];
  b = ptr[1];
  c = ptr[2];

  if (nDimA > 1) {
    grid_x = dimA[0];

    for (unsigned i = 1; i < nDimA; i++)
      total_lines *= dimA[i];
    grid_y = total_lines;

    for (unsigned i = 1; i < nDimB; i++)
      total_lines_b *= dimB[i];
  } else if (nDimA > 0) {
    grid_x = dimA[0];
    grid_y = 1;
  } else {
    throw std::runtime_error("Addtion is not possible with tensors without "
                             "any elements and dimensions zero.\n");
  }

  if (!isBroadCast) {
#pragma omp for
    for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
      unsigned idx = line_it * grid_x;
      for (unsigned i = 0; i + 4 <= grid_x; i += 4) {
        __m256d c_arr = _mm256_add_pd(
            _mm256_loadu_pd(reinterpret_cast<const double *>(a + i + idx)),
            _mm256_loadu_pd(reinterpret_cast<const double *>(b + i + idx)));

        _mm256_storeu_pd(reinterpret_cast<double *>(c + i + idx), c_arr);
      }

      // tackling remaining elements if any
      for (unsigned i = grid_x - (grid_x % 4); i < grid_x; i++) {
        ptr[2][idx + i] = ptr[1][idx + i] + ptr[0][idx + i];
      }
    }
  } else {
#pragma omp parallel for
    for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
      unsigned total_line_a = grid_y / dimA[nDimA - 1];
      unsigned total_line_b = total_lines_b / dimB[nDimB - 1];
      unsigned index_a = line_it;
      unsigned idx_b = 0;
      for (unsigned i = nDimA - 1; i > 0; i--) {
        unsigned indices = index_a / total_line_a;
        index_a -= indices * total_line_a;
        total_line_a /= dimA[i - 1];

        idx_b += indices * total_line_b * (dimB[i] != 1);
        total_line_b /= dimB[i - 1];
      }

      unsigned idx = line_it * grid_x;
      idx_b *= dimB[0];

      // unrolling the loop from 0 to n*4
      for (unsigned i = 0; i + 4 <= grid_x; i += 4) {
        if (dimB[0] != 1) {
          __m256d c_arr = _mm256_add_pd(
              _mm256_loadu_pd(reinterpret_cast<const double *>(a + idx + i)),
              _mm256_loadu_pd(reinterpret_cast<const double *>(b + idx_b + i)));

          _mm256_storeu_pd(reinterpret_cast<double *>(c + idx + i), c_arr);
        } else {
          __m256d c_arr = _mm256_add_pd(
              _mm256_loadu_pd(reinterpret_cast<const double *>(a + idx + i)),
              _mm256_set1_pd(static_cast<const double>(*(b + idx_b))));

          _mm256_storeu_pd(reinterpret_cast<double *>(c + idx + i), c_arr);
        }
      }

      // tackling remaining elements if any
      for (unsigned i = grid_x - (grid_x % 4); i < grid_x; i++) {
        ptr[2][idx + i] = ptr[0][idx + i] + ptr[1][idx_b + i * (dimB[0] != 1)];
      }
    }
  }
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
  unsigned vec_end = (n_elements / 4) * 4;
  omp_set_num_threads(std::thread::hardware_concurrency());
  LOG(INFO) << "avx256 kernel for sub is running....\n";

#pragma omp parallel for
  for (i = 0; i < vec_end; i += 4) {
    __m256d c_arr =
        _mm256_sub_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(a + i)),
                      _mm256_loadu_pd(reinterpret_cast<const double *>(b + i)));

    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }

  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = a[i] + b[i];
}

void avx2::avx2_sub_broadcast_f64(std::float64_t **ptr, const unsigned nDimA,
                                  const unsigned *dimA, const unsigned nDimB,
                                  const unsigned *dimB,
                                  const bool isBroadCast) {
  unsigned grid_x, grid_y;
  unsigned n_dim_A = nDimA;
  unsigned total_lines = 1;
  unsigned total_lines_b = 1;
  std::float64_t *a, *b, *c;
  a = ptr[0];
  b = ptr[1];
  c = ptr[2];

  if (nDimA > 1) {
    grid_x = dimA[0];

    for (unsigned i = 1; i < nDimA; i++)
      total_lines *= dimA[i];
    grid_y = total_lines;

    for (unsigned i = 1; i < nDimB; i++)
      total_lines_b *= dimB[i];
  } else if (nDimA > 0) {
    grid_x = dimA[0];
    grid_y = 1;
  } else {
    throw std::runtime_error("Addtion is not possible with tensors without "
                             "any elements and dimensions zero.\n");
  }

  if (!isBroadCast) {
#pragma omp for
    for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
      unsigned idx = line_it * grid_x;
      for (unsigned i = 0; i + 4 <= grid_x; i += 4) {
        __m256d c_arr = _mm256_sub_pd(
            _mm256_loadu_pd(reinterpret_cast<const double *>(a + i + idx)),
            _mm256_loadu_pd(reinterpret_cast<const double *>(b + i + idx)));

        _mm256_storeu_pd(reinterpret_cast<double *>(c + i + idx), c_arr);
      }

      // tackling remaining elements if any
      for (unsigned i = grid_x - (grid_x % 4); i < grid_x; i++) {
        ptr[2][idx + i] = ptr[0][idx + i] - ptr[1][idx + i];
      }
    }
  } else {
#pragma omp parallel for
    for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
      unsigned total_line_a = grid_y / dimA[nDimA - 1];
      unsigned total_line_b = total_lines_b / dimB[nDimB - 1];
      unsigned index_a = line_it;
      unsigned idx_b = 0;
      for (unsigned i = nDimA - 1; i > 0; i--) {
        unsigned indices = index_a / total_line_a;
        index_a -= indices * total_line_a;
        total_line_a /= dimA[i - 1];

        idx_b += indices * total_line_b * (dimB[i] != 1);
        total_line_b /= dimB[i - 1];
      }

      unsigned idx = line_it * grid_x;
      idx_b *= dimB[0];

      // unrolling the loop from 0 to n*4
      for (unsigned i = 0; i + 4 <= grid_x; i += 4) {
        if (dimB[0] != 1) {
          __m256d c_arr = _mm256_sub_pd(
              _mm256_loadu_pd(reinterpret_cast<const double *>(a + idx + i)),
              _mm256_loadu_pd(reinterpret_cast<const double *>(b + idx_b + i)));

          _mm256_storeu_pd(reinterpret_cast<double *>(c + idx + i), c_arr);
        } else {
          __m256d c_arr = _mm256_sub_pd(
              _mm256_loadu_pd(reinterpret_cast<const double *>(a + idx + i)),
              _mm256_set1_pd(static_cast<const double>(*(b + idx_b))));

          _mm256_storeu_pd(reinterpret_cast<double *>(c + idx + i), c_arr);
        }
      }

      // tackling remaining elements if any
      for (unsigned i = grid_x - (grid_x % 4); i < grid_x; i++) {
        ptr[2][idx + i] = ptr[0][idx + i] - ptr[1][idx_b + i * (dimB[0] != 1)];
      }
    }
  }
}

void avx2::avx2_mul_f64(std::float64_t **ptr, const unsigned *arr) {
  std::float64_t *a, *b, *c;
  unsigned i, m_size, n_size, n_elements;
  a = ptr[0];
  b = ptr[1];
  c = ptr[2];

  m_size = arr[0];
  n_size = arr[1];

  n_elements = m_size * n_size;
  unsigned vec_end = (n_elements / 4) * 4;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for mul is running....\n";

#pragma omp parallel for
  for (i = 0; i < vec_end; i += 4) {
    __m256d c_arr =
        _mm256_mul_pd(_mm256_loadu_pd(reinterpret_cast<const double *>(a + i)),
                      _mm256_loadu_pd(reinterpret_cast<const double *>(b + i)));

    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }

  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = a[i] * b[i];
}

void avx2::avx2_mul_broadcast_f64(std::float64_t **ptr, const unsigned nDimA,
                                  const unsigned *dimA, const unsigned nDimB,
                                  const unsigned *dimB,
                                  const bool isBroadCast) {
  unsigned grid_x, grid_y;
  unsigned n_dim_A = nDimA;
  unsigned total_lines = 1;
  unsigned total_lines_b = 1;
  std::float64_t *a, *b, *c;
  a = ptr[0];
  b = ptr[1];
  c = ptr[2];

  if (nDimA > 1) {
    grid_x = dimA[0];

    for (unsigned i = 1; i < nDimA; i++)
      total_lines *= dimA[i];
    grid_y = total_lines;

    for (unsigned i = 1; i < nDimB; i++)
      total_lines_b *= dimB[i];
  } else if (nDimA > 0) {
    grid_x = dimA[0];
    grid_y = 1;
  } else {
    throw std::runtime_error("Addtion is not possible with tensors without "
                             "any elements and dimensions zero.\n");
  }

  if (!isBroadCast) {
#pragma omp for
    for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
      unsigned idx = line_it * grid_x;
      for (unsigned i = 0; i + 4 <= grid_x; i += 4) {
        __m256d c_arr = _mm256_mul_pd(
            _mm256_loadu_pd(reinterpret_cast<const double *>(a + i + idx)),
            _mm256_loadu_pd(reinterpret_cast<const double *>(b + i + idx)));

        _mm256_storeu_pd(reinterpret_cast<double *>(c + i + idx), c_arr);
      }

      // tackling remaining elements if any
      for (unsigned i = grid_x - (grid_x % 4); i < grid_x; i++) {
        ptr[2][idx + i] = ptr[1][idx + i] * ptr[0][idx + i];
      }
    }
  } else {
#pragma omp parallel for
    for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
      unsigned total_line_a = grid_y / dimA[nDimA - 1];
      unsigned total_line_b = total_lines_b / dimB[nDimB - 1];
      unsigned index_a = line_it;
      unsigned idx_b = 0;
      for (unsigned i = nDimA - 1; i > 0; i--) {
        unsigned indices = index_a / total_line_a;
        index_a -= indices * total_line_a;
        total_line_a /= dimA[i - 1];

        idx_b += indices * total_line_b * (dimB[i] != 1);
        total_line_b /= dimB[i - 1];
      }

      unsigned idx = line_it * grid_x;
      idx_b *= dimB[0];

      // unrolling the loop from 0 to n*4
      for (unsigned i = 0; i + 4 <= grid_x; i += 4) {
        if (dimB[0] != 1) {
          __m256d c_arr = _mm256_mul_pd(
              _mm256_loadu_pd(reinterpret_cast<const double *>(a + idx + i)),
              _mm256_loadu_pd(reinterpret_cast<const double *>(b + idx_b + i)));

          _mm256_storeu_pd(reinterpret_cast<double *>(c + idx + i), c_arr);
        } else {
          __m256d c_arr = _mm256_mul_pd(
              _mm256_loadu_pd(reinterpret_cast<const double *>(a + idx + i)),
              _mm256_set1_pd(static_cast<const double>(*(b + idx_b))));

          _mm256_storeu_pd(reinterpret_cast<double *>(c + idx + i), c_arr);
        }
      }

      // tackling remaining elements if any
      for (unsigned i = grid_x - (grid_x % 4); i < grid_x; i++) {
        ptr[2][idx + i] = ptr[0][idx + i] * ptr[1][idx_b + i * (dimB[0] != 1)];
      }
    }
  }
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
  unsigned vec_end = (n_elements / 4) * 4;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for scale is running....\n";

#pragma omp parallel for
  for (i = 0; i < vec_end; i += 4) {
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
  unsigned vec_end = (n_elements / 4) * 4;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for sqrt is running....\n";
#pragma omp parallel for
  for (i = 0; i < vec_end; i += 4) {
    __m256d c_arr = _mm256_sqrt_pd(
        _mm256_loadu_pd(reinterpret_cast<const double *>(a + i)));
    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }
  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = std::sqrt(a[i]);
}

inline double horizontal_sum(__m256d v) {
  __m256d h = _mm256_hadd_pd(v, v);

  __m128d lo = _mm256_castpd256_pd128(h);
  __m128d hi = _mm256_extractf128_pd(h, 1);

  lo = _mm_add_pd(lo, hi);

  return _mm_cvtsd_f64(lo);
}

void avx2::avx2_reduce_sum_f64(std::float64_t *const *const ptr,
                               unsigned const *const arr) {

  std::float64_t *const input = ptr[0];
  std::float64_t *const output = ptr[1];

  unsigned inner_stride(1);
  unsigned outer_stride(1);
  unsigned no_of_dims = arr[0];
  unsigned axis = arr[no_of_dims + 1];
  unsigned axis_depth = arr[axis + 1];

  for (unsigned i = 1; i <= axis; i++)
    inner_stride *= arr[i];

  for (unsigned i = axis + 2; i <= no_of_dims; i++)
    outer_stride *= arr[i];

  unsigned no_of_line = inner_stride * outer_stride;

  if (!axis) {
#pragma omp parallel for
    for (unsigned line = 0; line < no_of_line; line++) {
      __m256d sum = _mm256_setzero_pd();
      unsigned i = 0;
      double result = 0.0;
      for (; i + 8 <= axis_depth; i += 8) {
        __m256d a = _mm256_loadu_pd(
            reinterpret_cast<const double *>(input + i + (line * axis_depth)));
        __m256d b = _mm256_loadu_pd(reinterpret_cast<const double *>(
            input + (i + 4) + (line * axis_depth)));

        sum = _mm256_add_pd(sum, a);
        sum = _mm256_add_pd(sum, b);
      }
      result = horizontal_sum(sum);

      for (; i < axis_depth; i++)
        result += input[i + line * axis_depth];
      output[line] = result;
    }
  } else {
    // clang-format off
#pragma omp parallel
    {
      // clang-format on
      for (unsigned line = 0; line + 4 <= no_of_line; line += 4) {
        unsigned line_x = line % inner_stride;
        unsigned line_y = line / inner_stride;
        unsigned base_index = line_x + line_y * inner_stride * axis_depth;
        unsigned output_index = line_x + line_y * inner_stride;

        __m256d sum = _mm256_loadu_pd(
            reinterpret_cast<const double *>(input + base_index));
        for (unsigned i = 1; i < axis_depth; i++) {
          sum = _mm256_add_pd(sum,
                              _mm256_loadu_pd(reinterpret_cast<const double *>(
                                  input + (base_index + i * inner_stride))));
        }
        _mm256_storeu_pd(reinterpret_cast<double *>(output + output_index),
                         sum);
      }
      // clang-format off
    }
    // clang-format on

    for (unsigned line = (no_of_line / 4) * 4; line < no_of_line; line++) {
      unsigned line_x = line % inner_stride;
      unsigned line_y = line / inner_stride;
      unsigned base_index = line_x + line_y * inner_stride * axis_depth;
      unsigned output_index = line_x + line_y * inner_stride;

      std::float64_t sum = input[base_index];
      for (unsigned i = 1; i < axis_depth; i++) {
        sum += input[base_index + i * inner_stride];
      }
      output[output_index] = sum;
    }
  }
}

void avx2::avx2_relu_f64(std::float64_t **ptr, unsigned const nDim,
                         unsigned const *arr) {
  std::float64_t *a, *c;
  unsigned i, m_size, n_size, total_plane, n_elements;
  a = ptr[0];
  c = ptr[1];

  m_size = arr[0];
  n_size = arr[1];

  total_plane = 1;
  if (nDim > 2)
    for (unsigned i = 2; i < nDim; i++)
      total_plane *= arr[i];

  n_elements = m_size * n_size * total_plane;
  unsigned vec_end = (n_elements / 4) * 4;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for relu is running....\n";
#pragma omp parallel for
  for (i = 0; i < vec_end; i += 4) {
    __m256d zero = _mm256_setzero_pd();
    __m256d c_arr = _mm256_max_pd(
        _mm256_loadu_pd(reinterpret_cast<const double *>(a + i)), zero);
    _mm256_storeu_pd(reinterpret_cast<double *>(c + i), c_arr);
  }
  for (i = n_elements - (n_elements % 4); i < n_elements; i++)
    c[i] = std::fmax(a[i], 0.0);
}

// scale "a" by 2^k, where k is a double vector (integer values stored as
// double)
static inline __m256d mul_pow2_pd(__m256d a, __m256d k_real) {
  alignas(32) double tmp[4];
  _mm256_storeu_pd(tmp, k_real); // dump vector to array

  int64_t i0 = (int64_t)tmp[0];
  int64_t i1 = (int64_t)tmp[1];
  int64_t i2 = (int64_t)tmp[2];
  int64_t i3 = (int64_t)tmp[3];

  uint64_t e0 = (uint64_t)(i0 + 1023) << 52;
  uint64_t e1 = (uint64_t)(i1 + 1023) << 52;
  uint64_t e2 = (uint64_t)(i2 + 1023) << 52;
  uint64_t e3 = (uint64_t)(i3 + 1023) << 52;

  alignas(32) uint64_t powbits[4] = {e0, e1, e2, e3};
  __m256d pow2k = _mm256_castsi256_pd(_mm256_load_si256((__m256i *)powbits));

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
  unsigned vec_end = (n_elements / 4) * 4;
  omp_set_num_threads(std::thread::hardware_concurrency());

  LOG(INFO) << "avx256 kernel for sigmoid is running....\n";
#pragma omp parallel for
  for (i = 0; i < vec_end; i += 4) {
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

/**
 * @brief AVX2 Micro Kernel: Softmax on tensor
 * @param ptr double pointer to input and output tensor index respectively
 * @param arr encoded unsigned array
 *            arr[0]: softmax axis
 *            arr[1]: no of dimensions for the tensors
 *            arr[2-n]: dimensions
 */
void avx2::avx2_softmax_f64(std::float64_t *const *const ptr,
                            unsigned const axis, unsigned const vector_length,
                            unsigned const inner_stride,
                            unsigned const outer_stride) {
  std::float64_t *A = ptr[0];
  std::float64_t *C = ptr[1];

  unsigned no_of_lines = outer_stride * inner_stride;

#pragma omp parallel for
  for (unsigned line = 0; line < no_of_lines; line++) {
    unsigned outer = line / inner_stride;
    unsigned inner = line % inner_stride;
    unsigned base = outer * vector_length * inner_stride + inner;

    std::float64_t max = A[base];
    for (unsigned i = 1; i < vector_length; i++) {
      std::float64_t value = A[base + i * inner_stride];
      if (max < value)
        max = value;
    }

    std::float64_t sum = 0.0;
    for (unsigned i = 0; i < vector_length; i++) {
      std::float64_t value = std::exp(A[base + i * inner_stride] - max);
      C[base + i * inner_stride] = value;
      sum += value;
    }

    for (unsigned i = 0; i < vector_length; i++)
      C[base + i * inner_stride] /= sum;
  }
}

void avx2::avx2_greater_than_zero_f64(std::float64_t *const *const ptr,
                                      unsigned int *const dims,
                                      const unsigned int nDims) {
  unsigned grid_x, grid_y;
  unsigned total_lines = 1;

  if (nDims > 1) {
    grid_x = dims[0];
    for (unsigned i = 1; i < nDims; i++)
      total_lines *= dims[i];
    grid_y = total_lines;
  } else if (nDims > 0) {
    grid_x = dims[0];
    grid_y = 1;
  } else {
    throw std::runtime_error(
        "operation: greater_than_zero is not possible with tensors without "
        "any elements and dimensions zero.\n");
  }

#pragma omp parallel for
  for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
    unsigned idx = line_it * grid_x;

    std::float64_t *in = ptr[0] + idx;
    std::float64_t *out = ptr[1] + idx;

    __m256d zero = _mm256_setzero_pd();
    __m256d one = _mm256_set1_pd(1.0);

    unsigned i = 0;

    // Process 4 doubles at a time
    for (; i + 4 <= grid_x; i += 4) {
      __m256d x = _mm256_loadu_pd(reinterpret_cast<const double *>(in + i));

      // mask = (x > 0)
      __m256d mask = _mm256_cmp_pd(x, zero, _CMP_GT_OQ);

      // Convert mask → 0.0 or 1.0
      __m256d result = _mm256_and_pd(mask, one);

      _mm256_storeu_pd(reinterpret_cast<double *>(out + i), result);
    }

    // Tail loop
    for (; i < grid_x; ++i) {
      out[i] = (in[i] > 0) ? 1.0 : 0.0;
    }
  }
}
