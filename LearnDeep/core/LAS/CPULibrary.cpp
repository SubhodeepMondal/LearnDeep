#include "CPULibrary.h"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <vector>

#define TILE_DOUBLE_X 8
#define TILE_DOUBLE_Y 16
#define TILE_FLOAT_X 16
#define TILE_FLOAT_Y 16
#define TILE_INT_X 16
#define TILE_INT_Y 16

void cpu::__matmul(std::float64_t **ptr, unsigned *arr) {
  // x output row size
  // y k row
  // z output column size
  // omp_set_num_threads(12);
  std::float64_t *A, *B, *C;

  A = ptr[0];
  B = ptr[1];
  C = ptr[2];
  unsigned x, y, z;
  x = arr[0];
  y = arr[1];
  z = arr[2];
  // std::cout << omp_get_max_threads() << "\n";
  memset(C, 0, sizeof(std::float64_t) * x * z);
#pragma omp parallel proc_bind(close)
  {
#pragma omp for
    for (int k = 0; k < z; k++) {
      for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
          C[i + k * x] += A[j + k * y] * B[i + j * x];
        }
      }
    }
  }
}

void cpu::__matmul_conventional(std::float64_t **ptr, unsigned *arr) {

  std::float64_t sum, *A, *B, *C;
  unsigned x, y, z;

  A = ptr[0];
  B = ptr[1];
  C = ptr[2];

  x = arr[0];
  y = arr[1];
  z = arr[2];
  // x output row size
  // y k row
  // z output column size
  // omp_set_num_threads(12);

  memset(C, 0, sizeof(std::float64_t) * x * z);
#pragma omp parallel for private(sum)
  for (int j = 0; j < z; j++)
    for (int i = 0; i < x; i++) {
      sum = 0;
      for (int k = 0; k < y; k++)
        sum += A[k + j * y] * B[i + k * x];
      C[i + j * x] = sum;
    }
}

void cpu::__melementwisemul(std::float64_t **ptr, const unsigned *arr) {
  std::float64_t *A, *B, *C;
  unsigned i, j, x, y, idx;

  A = ptr[0];
  B = ptr[1];
  C = ptr[2];

  x = arr[0];
  y = arr[1];

  // std::cout << x << " " << y << " In element wise mul!\n";

#pragma omp parallel for
  for (j = 0; j < y; j++)
    for (i = 0; i < x; i++) {
      idx = i + j * x;
      C[idx] = A[idx] * B[idx];
    }
}

void cpu::__mmul_broadcast(std::float64_t *const *const ptr,
                           const unsigned nDimA, const unsigned *dimA,
                           const unsigned nDimB, const unsigned *dimB,
                           const bool isBroadCast) {
  unsigned grid_x, grid_y;
  unsigned total_lines = 1;
  unsigned total_lines_b = 1;
  unsigned n_dim_A = nDimA;

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

  // omp_set_num_threads(1);

  if (!isBroadCast) {
    // clang-format off
#pragma omp parallel
{
      // clang-format on
      alignas(64) std::float64_t temp_output[8];
#pragma omp for
      for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
        unsigned idx = line_it * grid_x;
        for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
          temp_output[0] = ptr[1][idx + i] * ptr[0][idx + i];
          temp_output[1] = ptr[1][idx + i + 1] * ptr[0][idx + (i + 1)];
          temp_output[2] = ptr[1][idx + i + 2] * ptr[0][idx + (i + 2)];
          temp_output[3] = ptr[1][idx + i + 3] * ptr[0][idx + (i + 3)];
          temp_output[4] = ptr[1][idx + i + 4] * ptr[0][idx + (i + 4)];
          temp_output[5] = ptr[1][idx + i + 5] * ptr[0][idx + (i + 5)];
          temp_output[6] = ptr[1][idx + i + 6] * ptr[0][idx + (i + 6)];
          temp_output[7] = ptr[1][idx + i + 7] * ptr[0][idx + (i + 7)];

          std::memcpy(&ptr[2][idx + i], temp_output,
                      8 * sizeof(std::float64_t));
        }

        // tackling remaining elements if any
        for (unsigned i = grid_x - (grid_x % 8); i < grid_x; i++) {
          ptr[2][idx + i] = ptr[1][idx + i] * ptr[0][idx + i];
        }
        // }
      }
      // clang-format off
}
    // clang-format on
  } else {
    // clang-format off
#pragma omp parallel
{
      // clang-format on
      alignas(64) std::float64_t temp_output[8];
#pragma omp for
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

        // unrolling the loop from 0 to n*8
        for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
          temp_output[0] = ptr[0][idx + i] * ptr[1][idx_b + i * (dimB[0] != 1)];
          temp_output[1] =
              ptr[0][idx + i + 1] * ptr[1][idx_b + (i + 1) * (dimB[0] != 1)];
          temp_output[2] =
              ptr[0][idx + i + 2] * ptr[1][idx_b + (i + 2) * (dimB[0] != 1)];
          temp_output[3] =
              ptr[0][idx + i + 3] * ptr[1][idx_b + (i + 3) * (dimB[0] != 1)];
          temp_output[4] =
              ptr[0][idx + i + 4] * ptr[1][idx_b + (i + 4) * (dimB[0] != 1)];
          temp_output[5] =
              ptr[0][idx + i + 5] * ptr[1][idx_b + (i + 5) * (dimB[0] != 1)];
          temp_output[6] =
              ptr[0][idx + i + 6] * ptr[1][idx_b + (i + 6) * (dimB[0] != 1)];
          temp_output[7] =
              ptr[0][idx + i + 7] * ptr[1][idx_b + (i + 7) * (dimB[0] != 1)];

          std::memcpy(&ptr[2][idx + i], temp_output,
                      8 * sizeof(std::float64_t));
        }

        // working on the remaining elements if any
        for (unsigned i = grid_x - (grid_x % 8); i < grid_x; ++i) {
          ptr[2][idx + i] =
              ptr[0][idx + i] * ptr[1][idx_b + i * (dimB[0] != 1)];
        }
      }
      // clang-format off
}
    // clang-format on
  }
}

void cpu::__mscalermul(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *A, B, *C;
  unsigned x, y;

  A = ptr[0];
  B = ptr[1][0];
  C = ptr[2];

  x = arr[0];
  y = arr[1];

#pragma omp parallel for
  for (unsigned j = 0; j < y; j++)
    for (unsigned i = 0; i < x; i++)
      C[i + j * x] = B * A[i + j * x];
}

void cpu::__mscalermul_broadcast(std::float64_t *const *const ptr,
                                 const unsigned nDimA, const unsigned *dimA,
                                 const unsigned nDimB, const unsigned *dimB,
                                 const bool isBroadCast) {
  unsigned grid_x, grid_y;
  unsigned total_lines = 1;
  unsigned total_lines_b = 1;
  unsigned n_dim_A = nDimA;

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

  // omp_set_num_threads(1);

  if (!isBroadCast) {
    // clang-format off
#pragma omp parallel
{
      // clang-format on
      alignas(64) std::float64_t temp_output[8];
#pragma omp for
      for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
        unsigned idx = line_it * grid_x;
        for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
          temp_output[0] = ptr[1][idx + i] * ptr[0][idx + i];
          temp_output[1] = ptr[1][idx + i + 1] * ptr[0][idx + (i + 1)];
          temp_output[2] = ptr[1][idx + i + 2] * ptr[0][idx + (i + 2)];
          temp_output[3] = ptr[1][idx + i + 3] * ptr[0][idx + (i + 3)];
          temp_output[4] = ptr[1][idx + i + 4] * ptr[0][idx + (i + 4)];
          temp_output[5] = ptr[1][idx + i + 5] * ptr[0][idx + (i + 5)];
          temp_output[6] = ptr[1][idx + i + 6] * ptr[0][idx + (i + 6)];
          temp_output[7] = ptr[1][idx + i + 7] * ptr[0][idx + (i + 7)];

          std::memcpy(&ptr[2][idx + i], temp_output,
                      8 * sizeof(std::float64_t));
        }

        // tackling remaining elements if any
        for (unsigned i = grid_x - (grid_x % 8); i < grid_x; i++) {
          ptr[2][idx + i] = ptr[0][idx + i] - ptr[1][idx + i];
        }
        // }
      }
      // clang-format off
}
    // clang-format on
  } else {
    // clang-format off
#pragma omp parallel
{
      // clang-format on
      alignas(64) std::float64_t temp_output[8];
#pragma omp for
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

        // unrolling the loop from 0 to n*8
        for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
          temp_output[0] = ptr[0][idx + i] * ptr[1][idx_b + i * (dimB[0] != 1)];
          temp_output[1] =
              ptr[0][idx + i + 1] * ptr[1][idx_b + (i + 1) * (dimB[0] != 1)];
          temp_output[2] =
              ptr[0][idx + i + 2] * ptr[1][idx_b + (i + 2) * (dimB[0] != 1)];
          temp_output[3] =
              ptr[0][idx + i + 3] * ptr[1][idx_b + (i + 3) * (dimB[0] != 1)];
          temp_output[4] =
              ptr[0][idx + i + 4] * ptr[1][idx_b + (i + 4) * (dimB[0] != 1)];
          temp_output[5] =
              ptr[0][idx + i + 5] * ptr[1][idx_b + (i + 5) * (dimB[0] != 1)];
          temp_output[6] =
              ptr[0][idx + i + 6] * ptr[1][idx_b + (i + 6) * (dimB[0] != 1)];
          temp_output[7] =
              ptr[0][idx + i + 7] * ptr[1][idx_b + (i + 7) * (dimB[0] != 1)];

          std::memcpy(&ptr[2][idx + i], temp_output,
                      8 * sizeof(std::float64_t));
        }

        // working on the remaining elements if any
        for (unsigned i = grid_x - (grid_x % 8); i < grid_x; ++i) {
          ptr[2][idx + i] =
              ptr[0][idx + i] * ptr[1][idx_b + i * (dimB[0] != 1)];
        }
      }
      // clang-format off
}
    // clang-format on
  }
}

void cpu::__madd(std::float64_t **ptr, unsigned *a) {
  std::float64_t *inp_a, *inp_b, *out;
  unsigned x, y;

  inp_a = ptr[0];
  inp_b = ptr[1];
  out = ptr[2];

  x = a[0];
  y = a[1];

#pragma omp parallel for
  for (int j = 0; j < y; j++)
    for (int i = 0; i < x; i++)
      out[i + j * x] = inp_a[i + j * x] + inp_b[i + j * x];
}

void cpu::__madd_broadcast(std::float64_t *const *const ptr,
                           const unsigned nDimA, const unsigned *dimA,
                           const unsigned nDimB, const unsigned *dimB,
                           const bool isBroadCast) {
  unsigned grid_x, grid_y;
  unsigned total_lines = 1;
  unsigned total_lines_b = 1;
  unsigned n_dim_A = nDimA;

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

  // omp_set_num_threads(1);

  if (!isBroadCast) {
    // clang-format off
#pragma omp parallel 
{
    // clang-format on  
    alignas(64) std::float64_t temp_output[8];
    #pragma omp for
    for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
      unsigned idx = line_it * grid_x;
        for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
          temp_output[0] = ptr[1][idx + i] + ptr[0][idx + i];
          temp_output[1] = ptr[1][idx + i + 1] + ptr[0][idx + (i + 1)];
          temp_output[2] = ptr[1][idx + i + 2] + ptr[0][idx + (i + 2)];
          temp_output[3] = ptr[1][idx + i + 3] + ptr[0][idx + (i + 3)];
          temp_output[4] = ptr[1][idx + i + 4] + ptr[0][idx + (i + 4)];
          temp_output[5] = ptr[1][idx + i + 5] + ptr[0][idx + (i + 5)];
          temp_output[6] = ptr[1][idx + i + 6] + ptr[0][idx + (i + 6)];
          temp_output[7] = ptr[1][idx + i + 7] + ptr[0][idx + (i + 7)];

          std::memcpy(&ptr[2][idx + i], temp_output, 8 * sizeof(std::float64_t));
        }

        // tackling remaining elements if any
        for (unsigned i = grid_x - (grid_x % 8); i < grid_x; i++) {
          ptr[2][idx + i] = ptr[0][idx + i] + ptr[1][idx + i];
        }
      // }
    }
    // clang-format off
}
    // clang-format on
  } else {
    // clang-format off
#pragma omp parallel 
{
      // clang-format on
      alignas(64) std::float64_t temp_output[8];
#pragma omp for
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

        // unrolling the loop from 0 to n*8
        for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
          temp_output[0] = ptr[0][idx + i] + ptr[1][idx_b + i * (dimB[0] != 1)];
          temp_output[1] =
              ptr[0][idx + i + 1] + ptr[1][idx_b + (i + 1) * (dimB[0] != 1)];
          temp_output[2] =
              ptr[0][idx + i + 2] + ptr[1][idx_b + (i + 2) * (dimB[0] != 1)];
          temp_output[3] =
              ptr[0][idx + i + 3] + ptr[1][idx_b + (i + 3) * (dimB[0] != 1)];
          temp_output[4] =
              ptr[0][idx + i + 4] + ptr[1][idx_b + (i + 4) * (dimB[0] != 1)];
          temp_output[5] =
              ptr[0][idx + i + 5] + ptr[1][idx_b + (i + 5) * (dimB[0] != 1)];
          temp_output[6] =
              ptr[0][idx + i + 6] + ptr[1][idx_b + (i + 6) * (dimB[0] != 1)];
          temp_output[7] =
              ptr[0][idx + i + 7] + ptr[1][idx_b + (i + 7) * (dimB[0] != 1)];

          std::memcpy(&ptr[2][idx + i], temp_output,
                      8 * sizeof(std::float64_t));
        }

        // working on the remaining elements if any
        for (unsigned i = grid_x - (grid_x % 8); i < grid_x; ++i) {
          ptr[2][idx + i] =
              ptr[0][idx + i] + ptr[1][idx_b + i * (dimB[0] != 1)];
        }
      }
      // clang-format off
}
    // clang-format on
  }
}

void cpu::__msub(std::float64_t **ptr, unsigned *a) {
  std::float64_t *inp_a, *inp_b, *out;
  unsigned x, y;
  inp_a = ptr[0];
  inp_b = ptr[1];
  out = ptr[2];

  x = a[0];
  y = a[1];

#pragma omp parallel for
  for (int i = 0; i < y; i++)
    for (int j = 0; j < x; j++)
      out[j + i * x] = inp_a[j + i * x] - inp_b[j + i * x];
}

void cpu::__msub_broadcast(std::float64_t *const *const ptr,
                           const unsigned nDimA, const unsigned *dimA,
                           const unsigned nDimB, const unsigned *dimB,
                           const bool isBroadCast) {
  unsigned grid_x, grid_y;
  unsigned total_lines = 1;
  unsigned total_lines_b = 1;
  unsigned n_dim_A = nDimA;

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

  // omp_set_num_threads(1);

  if (!isBroadCast) {
    // clang-format off
#pragma omp parallel
{
      // clang-format on
      alignas(64) std::float64_t temp_output[8];
#pragma omp for
      for (unsigned line_it = 0; line_it < grid_y; ++line_it) {
        unsigned idx = line_it * grid_x;
        for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
          temp_output[0] = ptr[0][idx + i] - ptr[1][idx + i];
          temp_output[1] = ptr[0][idx + (i + 1)] - ptr[1][idx + i + 1];
          temp_output[2] = ptr[0][idx + (i + 2)] - ptr[1][idx + i + 2];
          temp_output[3] = ptr[0][idx + (i + 3)] - ptr[1][idx + i + 3];
          temp_output[4] = ptr[0][idx + (i + 4)] - ptr[1][idx + i + 4];
          temp_output[5] = ptr[0][idx + (i + 5)] - ptr[1][idx + i + 5];
          temp_output[6] = ptr[0][idx + (i + 6)] - ptr[1][idx + i + 6];
          temp_output[7] = ptr[0][idx + (i + 7)] - ptr[1][idx + i + 7];

          std::memcpy(&ptr[2][idx + i], temp_output,
                      8 * sizeof(std::float64_t));
        }

        // tackling remaining elements if any
        for (unsigned i = grid_x - (grid_x % 8); i < grid_x; i++) {
          ptr[2][idx + i] = ptr[0][idx + i] - ptr[1][idx + i];
        }
        // }
      }
      // clang-format off
}
    // clang-format on
  } else {
    // clang-format off
#pragma omp parallel
{
      // clang-format on
      alignas(64) std::float64_t temp_output[8];
#pragma omp for
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

        // unrolling the loop from 0 to n*8
        for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
          temp_output[0] = ptr[0][idx + i] - ptr[1][idx_b + i * (dimB[0] != 1)];
          temp_output[1] =
              ptr[0][idx + i + 1] - ptr[1][idx_b + (i + 1) * (dimB[0] != 1)];
          temp_output[2] =
              ptr[0][idx + i + 2] - ptr[1][idx_b + (i + 2) * (dimB[0] != 1)];
          temp_output[3] =
              ptr[0][idx + i + 3] - ptr[1][idx_b + (i + 3) * (dimB[0] != 1)];
          temp_output[4] =
              ptr[0][idx + i + 4] - ptr[1][idx_b + (i + 4) * (dimB[0] != 1)];
          temp_output[5] =
              ptr[0][idx + i + 5] - ptr[1][idx_b + (i + 5) * (dimB[0] != 1)];
          temp_output[6] =
              ptr[0][idx + i + 6] - ptr[1][idx_b + (i + 6) * (dimB[0] != 1)];
          temp_output[7] =
              ptr[0][idx + i + 7] - ptr[1][idx_b + (i + 7) * (dimB[0] != 1)];

          std::memcpy(&ptr[2][idx + i], temp_output,
                      8 * sizeof(std::float64_t));
        }

        // working on the remaining elements if any
        for (unsigned i = grid_x - (grid_x % 8); i < grid_x; ++i) {
          ptr[2][idx + i] =
              ptr[0][idx + i] - ptr[1][idx_b + i * (dimB[0] != 1)];
        }
      }
      // clang-format off
}
    // clang-format on
  }
}

void cpu::__mrollingsum(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *inp, *output;
  unsigned axis, x, y, z;
  unsigned i, j, k, sum = 0;

  inp = ptr[0];
  output = ptr[1];

  axis = arr[0];
  x = arr[1];
  y = arr[2];
  z = arr[3];

  switch (axis) {
  case 0: {
    for (j = 0; j < z; j++)
      for (i = 0; i < y; i++) {
        sum = 0;
        for (k = 0; k < x; k++)
          sum += inp[k + i * x + j * x * y];
        output[i + j * x] = sum;
      }
    break;
  }
  default:
    break;
  }
}

void cpu::__mtranspose(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *A, *B;
  unsigned x, y;

  A = ptr[0];
  B = ptr[1];

  x = arr[0];
  y = arr[1];

#pragma omp parallel for collapse(2) schedule(static)
  for (int j = 0; j < y; j++) {
    for (int i = 0; i < x; i++)
      B[i + j * x] = A[j + i * y];
  }
}

void cpu::__mtiled_transpose(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *A, *B;
  unsigned x, y;
  // std::float64_t tile[TILE_DOUBLE_Y][TILE_DOUBLE_X];

  A = ptr[0];
  B = ptr[1];
  x = arr[0];
  y = arr[1];
#pragma omp parallel for collapse(2) schedule(static)
  for (unsigned idx_j = 0; idx_j < y / TILE_DOUBLE_Y; idx_j++) {
    for (unsigned idx_i = 0; idx_i < x / TILE_DOUBLE_X; idx_i++) {

      alignas(64) double tile[TILE_DOUBLE_Y]
                             [TILE_DOUBLE_X]; // row-major: tile[row][col]
      for (unsigned i = 0; i < TILE_DOUBLE_X; i++) {
        for (unsigned j = 0; j < TILE_DOUBLE_Y; j++) {
          tile[j][i] =
              A[(i + idx_i * TILE_DOUBLE_X) + (j + idx_j * TILE_DOUBLE_Y) * x];
        }
      }
      for (unsigned i = 0; i < TILE_DOUBLE_X; i++) {
        for (unsigned j = 0; j < TILE_DOUBLE_Y; j++) {
          B[(j + idx_j * TILE_DOUBLE_Y) + (i + idx_i * TILE_DOUBLE_X) * y] =
              tile[j][i];
        }
      }
    }
  }

  for (unsigned j = y - (y % TILE_DOUBLE_Y); j < y; j++)
    for (unsigned i = 0; i < x - (x % TILE_DOUBLE_X); i++)
      B[j + i * y] = A[i + j * x];

  for (unsigned j = 0; j < y - (y % TILE_DOUBLE_Y); j++)
    for (unsigned i = x - (x % TILE_DOUBLE_X); i < x; i++)
      B[j + i * y] = A[i + j * x];

  for (unsigned j = y - (y % TILE_DOUBLE_Y); j < y; j++)
    for (unsigned i = x - (x % TILE_DOUBLE_X); i < x; i++)
      B[j + i * y] = A[i + j * x];
}

void cpu::__msqrt(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *A, *C;
  unsigned x, y;

  A = ptr[0];
  C = ptr[1];

  x = arr[0];
  y = arr[1];
#pragma omp parallel for
  for (unsigned j = 0; j < y; j++)
    for (unsigned i = 0; i < x; i++)
      C[i + j * x] = std::sqrt(A[i + j * x]);
}

void cpu::__mreducesum(std::float64_t *const *const ptr,
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
      std::float64_t sum = 0;
      unsigned i = 0;
      for (; i + 8 <= axis_depth; i += 8) {
        sum += input[i + line * axis_depth];
        sum += input[(i + 1) + line * axis_depth];
        sum += input[(i + 2) + line * axis_depth];
        sum += input[(i + 3) + line * axis_depth];
        sum += input[(i + 4) + line * axis_depth];
        sum += input[(i + 5) + line * axis_depth];
        sum += input[(i + 6) + line * axis_depth];
        sum += input[(i + 7) + line * axis_depth];
      }

      for (; i < axis_depth; i++)
        sum += input[i + line * axis_depth];

      output[line] = sum;
    }
  } else {
#pragma omp parallel for schedule(static, 1)
    for (unsigned line = 0; line < no_of_line; line++) {
      std::float64_t sum = 0.0;
      unsigned line_x = line % inner_stride;
      unsigned line_y = line / inner_stride;
      unsigned base_index = line_x + line_y * inner_stride * axis_depth;
      unsigned output_index = line_y * inner_stride + line_x;

      unsigned i = 0;
      for (; i + 8 <= axis_depth; i += 8) {
        sum += input[base_index + i * inner_stride];
        sum += input[base_index + (i + 1) * inner_stride];
        sum += input[base_index + (i + 2) * inner_stride];
        sum += input[base_index + (i + 3) * inner_stride];
        sum += input[base_index + (i + 4) * inner_stride];
        sum += input[base_index + (i + 5) * inner_stride];
        sum += input[base_index + (i + 6) * inner_stride];
        sum += input[base_index + (i + 7) * inner_stride];
      }
      for (; i < axis_depth; i++)
        sum += input[base_index + i * inner_stride];
      output[output_index] = sum;
    }
  }
}

void cpu::__mrelu(std::float64_t **ptr, unsigned const *arr) {
  std::float64_t *A, *C;
  unsigned x, y;

  A = ptr[0];
  C = ptr[1];

  x = arr[0];
  y = arr[1];
#pragma omp parallel for
  for (unsigned j = 0; j < y; j++)
    for (unsigned i = 0; i < x; i++)
      C[i + j * x] = (A[i + j * x] > 0) ? A[i + j * x] : 0;
}

void cpu::__msigmoid(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *A, *C;
  unsigned x, y;

  A = ptr[0];
  C = ptr[1];

  x = arr[0];
  y = arr[1];
#pragma omp parallel for
  for (unsigned j = 0; j < y; j++)
    for (unsigned i = 0; i < x; i++)
      C[i + j * x] = 1 / (1 + std::exp(-A[i + j * x]));
}

inline std::float64_t findMax(std::float64_t *const ptr, unsigned const n) {
  std::float64_t max = 1e-10;
  for (unsigned i = 0; n; i++)
    if (max < ptr[i])
      max = ptr[i];

  return max;
}

inline void reduceByMax(std::float64_t *const ptr_A,
                        std::float64_t *const ptr_B, unsigned const max,
                        unsigned const n) {
  for (unsigned i = 0; i < n; i++)
    ptr_B[i] = ptr_A[i] - max;
}

inline std::float64_t sumofExponents(std::float64_t *const ptr,
                                     unsigned const n) {
  std::float64_t sum = 0;
  for (unsigned i = 0; i < n; i++)
    sum += std::exp(ptr[i]);
  return sum;
}

inline void getSoftMaxOnRow(std::float64_t *const ptr, unsigned const n,
                            std::float64_t const sum) {

  for (unsigned i = 0; i < n; i++)
    ptr[i] = std::exp(ptr[i]) / sum;
}

/**
 * @brief Micro Kernel: Softmax on tensor
 * @param ptr double pointer to input and output tensor index respectively
 * @param arr encoded unsigned array
 *            arr[0]: softmax axis
 *            arr[1]: no of dimensions for the tensors
 *            arr[2-n]: dimensions
 */
void cpu::__msoftmax(std::float64_t *const *const ptr, unsigned *const arr) {
  std::float64_t *A, *C;
  unsigned x, y;

  A = ptr[0];
  C = ptr[1];

  unsigned axis = arr[0];
  unsigned dims = arr[1];

  if (!axis) {
    unsigned no_of_lines = 1;
    for (unsigned i = 1; i < arr[1]; i++)
      no_of_lines *= arr[i];

#pragma omp parallel for
    for (unsigned j = 0; j < no_of_lines; j++) {
      std::float64_t max = findMax(A + j * arr[axis + 2], arr[axis + 2]);
      reduceByMax(A + j * arr[1], C + j * arr[axis + 2], arr[axis + 2], max);
      std::float64_t sum = sumofExponents(C + j * arr[axis + 2], arr[axis + 2]);
      getSoftMaxOnRow(C + j * arr[1], arr[1], sum);
    }
  } else {
    unsigned no_of_lines = 1;
    unsigned stride = 1;
    for (unsigned i = 0; i < arr[1]; i++) {
      if (i != axis)
        no_of_lines *= arr[i];
      if (i < axis)
        stride *= arr[i];
    }
    // clang-format off
#pragma omp parallel
{
      // clang-format on
      thread_local std::vector<std::float64_t> line_in;
      thread_local std::vector<std::float64_t> line_out;
      line_in.resize(arr[axis + 2]);
      line_out.resize(arr[axis + 2]);
#pragma omp for
      for (unsigned j = 0; j < no_of_lines; j++) {
        for (unsigned i = 0; i < arr[axis + 2]; i++)
          line_in[i] = A[i * stride + j];

        std::float64_t max = findMax(line_in.data(), arr[axis + 2]);
        reduceByMax(line_in.data(), line_out.data(), max, arr[axis + 2]);
        std::float64_t sum = sumofExponents(line_out.data(), arr[axis + 2]);
        getSoftMaxOnRow(line_out.data(), arr[axis + 2], sum);

        for (unsigned i = 0; i < arr[axis + 2]; i++)
          C[i * stride + j] = line_out[i];
      }
      // clang-format off
      }
    // clang-format on
  }
}

void cpu::__mgreaterthanzero(std::float64_t *const *const ptr,
                             unsigned const *dims, unsigned const nDims) {
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

    for (unsigned i = 0; i + 8 <= grid_x; i += 8) {
      out[i] = (in[i] > 0);
      out[i + 1] = (in[i + 1] > 0);
      out[i + 2] = (in[i + 2] > 0);
      out[i + 3] = (in[i + 3] > 0);
      out[i + 4] = (in[i + 4] > 0);
      out[i + 5] = (in[i + 5] > 0);
      out[i + 6] = (in[i + 6] > 0);
      out[i + 7] = (in[i + 7] > 0);
    }

    for (unsigned i = grid_x - (grid_x % 8); i < grid_x; i++) {
      out[i] = (in[i] > 0);
    }
  }
}
