#include "CPULibrary.h"
#include <cmath>
#include <cstring>
#include <omp.h>

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

void cpu::__melementwisemul(std::float64_t **ptr, unsigned *arr) {
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
  std::float64_t **temp;

  A = ptr[0];
  B = ptr[1];

  x = arr[0];
  y = arr[1];

  temp = new std::float64_t *[y];
  for (int j = 0; j < y; j++) {
    temp[j] = new std::float64_t[x];
    for (int i = 0; i < x; i++)

      temp[j][i] = A[j + i * y];
  }

  for (int j = 0; j < y; j++) {
    for (int i = 0; i < x; i++)
      B[i + j * x] = temp[j][i];

    delete[] temp[j];
  }

  delete[] temp;
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

void cpu::__mrelu(std::float64_t **ptr, unsigned *arr) {
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

void cpu::__msoftmax(std::float64_t **ptr, unsigned *arr) {
  std::float64_t *A, *C;
  unsigned x, y;

  A = ptr[0];
  C = ptr[1];

  x = arr[0];
  y = arr[1];
#pragma omp parallel for
  for (unsigned j = 0; j < y; j++) {
    std::float64_t sum = 0;
    for (unsigned i = 0; i < x; i++) {
      C[i + j * x] = std::exp(A[i + j * x]);
      sum += C[i + j * x];
    }
    for (unsigned i = 0; i < x; i++)
      C[i + j * x] = C[i + j * x] / sum;
  }
}