#include "gpu_micro_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE_DOUBLE 16
extern __shared__ double global_shared_mem_d[];

__global__ void gpu_kernel::printData(double *a, unsigned x, unsigned y,
                                      unsigned z) {
  int i, j, k;
  for (i = 0; i < z; i++)
    for (j = 0; j < y; j++)
      for (k = 0; k < x; k++) {
        if (k == x - 1)
          if (j == y - 1)
            printf(" %lf\n\n", a[k + j * x + i * x * y]);
          else
            printf(" %lf\n", a[k + j * x + i * x * y]);
        else
          printf(" %lf", a[k + j * x + i * x * y]);
      }
}

__global__ void gpu_kernel::print(double *a) { printf("%.6lf", *(a)); }

__global__ void gpu_kernel::cudaDotMul(double *a, double *b, double *c, int x,
                                       int y, int a_m, int a_n, int b_m,
                                       int b_n) {
  int n = b_m;
  int ind, i;
  double val;
  unsigned mask = 0xffffffff;

  ind = (threadIdx.x + x * b_m) + (y * a_m * b_m);
  if (threadIdx.x < a_n) {
    c[ind] = a[threadIdx.x + x * a_n] * b[threadIdx.x * b_n + y];
  } else
    c[ind] = 0.0f;

  __syncthreads();

  if (threadIdx.x < a_n) {
    for (i = n / 2; i > 0; i /= 2) {
      val = c[ind];
      val = __shfl_down_sync(mask, val, i);
      c[ind] += val;
    }
  }
}

__global__ void gpu_kernel::cudaMatrixMulMultiParallesied(double *a, double *b,
                                                          double *c, double *d,
                                                          int a_m, int a_n,
                                                          int b_m, int b_n) {
  int ix, iy, rowDim;
  ix = threadIdx.x + blockIdx.x * blockDim.x;
  iy = threadIdx.y + blockIdx.y * blockDim.y;
  rowDim = b_m;
  d[ix + iy * a_m] = 0;
  if (ix < a_m && iy < b_n) {
    cudaDotMul<<<1, rowDim>>>(a, b, c, ix, iy, a_m, a_n, b_m, b_n);
    d[ix + iy * a_m] += c[ix * b_m * a_m + iy * b_m];
  }
}

__global__ void gpu_kernel::cudaRollingSum(double *a) {
  int n = blockDim.x;
  int ind, i, k;
  double val;
  ind = threadIdx.x;
  unsigned x = 0xffffffff;

  k = n;
  if (threadIdx.x < n) {
    for (i = n / 2; i > 0; i /= 2) {
      val = a[ind];
      val = __shfl_down_sync(x, val, i);
      a[ind] += val;
      k /= 2;
    }
  }
}

__device__ double gpu_kernel::cudaSubDotMul(double *a, double *b, int a_m,
                                            int a_n, int b_m, int b_n, int n) {
  double sum = 0;

  for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

__global__ void gpu_kernel::cudaSubMul(double *a, double *b, double *d, int a_m,
                                       int a_n, int b_m, int b_n, int i,
                                       int j) {
  __shared__ double Y_shared[32][33];

  double val;

  int Ai, Bi, Bj, n;

  Bi = j + threadIdx.x;
  Bj = i + threadIdx.y;
  Ai = i + (threadIdx.y + blockDim.y * blockIdx.y) * a_n;
  (a_n - i) >= 32 ? n = 32 : n = (a_n - i);

  if (Bi < b_n && Bj < b_m) {
    Y_shared[threadIdx.x][threadIdx.y] = b[Bi + Bj * b_n];
  }
  __syncthreads();

  if (Bi < b_n && (threadIdx.y + blockDim.y * blockIdx.y) < a_m) {

    val = cudaSubDotMul((a + Ai), Y_shared[threadIdx.x], a_m, a_n, b_m, b_n, n);
    __syncthreads();

    d[Bi + (threadIdx.y + blockDim.y * blockIdx.y) * b_n] += val;
  }
}

__global__ void gpu_kernel::cudaMatrixMul(double *a, double *b, double *d,
                                          int a_m, int a_n, int b_m, int b_n,
                                          int i, int j) {
  __shared__ double Y_shared[32][33];
  __shared__ double X_shared[32][32];

  double val;

  int Ai, Bi, Bj, n, idx_Ai;

  Bi = j + threadIdx.x;
  Bj = i + threadIdx.y;
  Ai = i + (threadIdx.y + blockDim.y * blockIdx.y) * a_n;
  idx_Ai = (threadIdx.y + blockDim.y * blockIdx.y);

  (a_n - i) >= 32 ? n = 32 : n = (a_n - i);

  if (Bi < b_n && Bj < b_m) {
    Y_shared[threadIdx.x][threadIdx.y] = b[Bi + Bj * b_n];
  }
  if (idx_Ai < a_m && threadIdx.x < a_n) {
    X_shared[threadIdx.y][threadIdx.x] = a[Ai + threadIdx.x];
  }
  __syncthreads();

  if (Bi < b_n && (threadIdx.y + blockDim.y * blockIdx.y) < a_m) {

    val = cudaSubDotMul(X_shared[threadIdx.y], Y_shared[threadIdx.x], a_m, a_n,
                        b_m, b_n, n);
    __syncthreads();

    d[Bi + (threadIdx.y + blockDim.y * blockIdx.y) * b_n] += val;
  }
}

__global__ void
gpu_kernel::matrixSum(double *const input_1, double *const input_2,
                      double *const output, const unsigned x_axis_dim,
                      const unsigned y_axis_dim, const unsigned plane_count) {
  unsigned idx_x, idx_y, idx_z;
  idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
  idx_y = threadIdx.y + (blockDim.y * blockIdx.y);
  idx_z = threadIdx.z + (blockDim.z * blockIdx.z);
  size_t index = idx_x + idx_y * x_axis_dim + idx_z * x_axis_dim * y_axis_dim;

  if (idx_x < x_axis_dim && idx_y < y_axis_dim && idx_z < plane_count)
    output[index] = input_1[index] + input_2[index];
}

__global__ void gpu_kernel::matrixSumBroadCast(
    double *const input_1, double *const input_2, double *const output,
    const unsigned nDimInput_1, const unsigned *const dimInput_1,
    const unsigned nDimInput_2, const unsigned *const dimInput_2,
    const unsigned plane_count, const unsigned plane_cout_b) {

  size_t idx_x = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t idx_y = threadIdx.y + (blockIdx.y * blockDim.y);
  size_t idx_z = threadIdx.z + (blockIdx.z * blockDim.z);

  __shared__ size_t x_axis_inp_1, y_axis_inp_1;
  __shared__ size_t x_axis_inp_2, y_axis_inp_2;
  __shared__ size_t a_plane_count, b_plane_count;
  __shared__ size_t idx_a, idx_b;
  __shared__ size_t i;

  __shared__ unsigned indices;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    x_axis_inp_1 = dimInput_1[0];
    y_axis_inp_1 = (nDimInput_1 > 1) ? dimInput_1[1] : 1;

    x_axis_inp_2 = dimInput_2[0];
    y_axis_inp_2 = (nDimInput_2 > 1) ? dimInput_2[1] : 1;

    a_plane_count = plane_count / dimInput_1[nDimInput_1 - 1];
    b_plane_count = plane_cout_b / dimInput_2[nDimInput_2 - 1];
    idx_a = idx_z;
    idx_b = 0;
    for (i = nDimInput_1 - 1; i > 1; i--) {
      indices = idx_a / a_plane_count;
      idx_a -= indices * a_plane_count;
      a_plane_count /= dimInput_1[i - 1];

      idx_b += indices * b_plane_count * (dimInput_2[i] != 1);
      b_plane_count /= dimInput_2[i - 1];
    }
    idx_b *= x_axis_inp_2 * y_axis_inp_2;
  }
  __syncthreads();

  size_t index_b = idx_b;

  size_t index =
      idx_x + idx_y * x_axis_inp_1 + idx_z * x_axis_inp_1 * y_axis_inp_1;
  index_b +=
      idx_x * (x_axis_inp_2 != 1) + idx_y * (y_axis_inp_2 != 1) * x_axis_inp_2;
  if (idx_x < x_axis_inp_1 && idx_y < y_axis_inp_1 && idx_z < plane_count) {
    output[index] = input_1[index] + input_2[index_b];
  }
}

__global__ void gpu_kernel::matrixMulBroadCast(
    double *const input_1, double *const input_2, double *const output,
    const unsigned nDimInput_1, const unsigned *const dimInput_1,
    const unsigned nDimInput_2, const unsigned *const dimInput_2,
    const unsigned plane_count, const unsigned plane_cout_b) {

  size_t idx_x = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t idx_y = threadIdx.y + (blockIdx.y * blockDim.y);
  size_t idx_z = threadIdx.z + (blockIdx.z * blockDim.z);

  __shared__ size_t x_axis_inp_1, y_axis_inp_1;
  __shared__ size_t x_axis_inp_2, y_axis_inp_2;
  __shared__ size_t a_plane_count, b_plane_count;
  __shared__ size_t idx_a, idx_b;
  __shared__ size_t i;

  __shared__ unsigned indices;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    x_axis_inp_1 = dimInput_1[0];
    y_axis_inp_1 = (nDimInput_1 > 1) ? dimInput_1[1] : 1;

    x_axis_inp_2 = dimInput_2[0];
    y_axis_inp_2 = (nDimInput_2 > 1) ? dimInput_2[1] : 1;

    a_plane_count = plane_count / dimInput_1[nDimInput_1 - 1];
    b_plane_count = plane_cout_b / dimInput_2[nDimInput_2 - 1];
    idx_a = idx_z;
    idx_b = 0;
    for (i = nDimInput_1 - 1; i > 1; i--) {
      indices = idx_a / a_plane_count;
      idx_a -= indices * a_plane_count;
      a_plane_count /= dimInput_1[i - 1];

      idx_b += indices * b_plane_count * (dimInput_2[i] != 1);
      b_plane_count /= dimInput_2[i - 1];
    }
    idx_b *= x_axis_inp_2 * y_axis_inp_2;
  }
  __syncthreads();

  size_t index_b = idx_b;

  size_t index =
      idx_x + idx_y * x_axis_inp_1 + idx_z * x_axis_inp_1 * y_axis_inp_1;
  index_b +=
      idx_x * (x_axis_inp_2 != 1) + idx_y * (y_axis_inp_2 != 1) * x_axis_inp_2;
  if (idx_x < x_axis_inp_1 && idx_y < y_axis_inp_1 && idx_z < plane_count) {
    output[index] = input_1[index] * input_2[index_b];
  }
}

__global__ void gpu_kernel::matrixHadamardMul(double *const input_1,
                                              double *const input_2,
                                              double *const output,
                                              const unsigned x_axis_dim,
                                              const unsigned y_axis_dim,
                                              const unsigned plane_count) {
  unsigned idx_x, idx_y, idx_z;
  idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
  idx_y = threadIdx.y + (blockDim.y * blockIdx.y);
  idx_z = threadIdx.z + (blockDim.z * blockIdx.z);
  size_t index = idx_x + idx_y * x_axis_dim + idx_z * x_axis_dim * y_axis_dim;

  if (idx_x < x_axis_dim && idx_y < y_axis_dim && idx_z < plane_count)
    output[index] = input_1[index] * input_2[index];
}

__global__ void gpu_kernel::matrixResuffledMul(double *a, double *b, double *c,
                                               unsigned x, unsigned y,
                                               unsigned z) {
  // x output row axis
  // y output column axis
  // z collapsing axis
  unsigned id_x, id_y;
  unsigned lin_idx_a, lin_idx_b, lin_idx_c;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx_c = id_x + id_y * x;
  c[lin_idx_c] = 0.0;
  for (int i = 0; i < z; i++) {
    if (id_x < x && id_y < y) {
      lin_idx_a = i + id_y * z;
      lin_idx_b = id_x + i * x;
      c[lin_idx_c] += a[lin_idx_a] * b[lin_idx_b];
    }
  }
}

__global__ void gpu_kernel::matrixTiledMul(double *a, double *b, double *c,
                                           unsigned x, unsigned y, unsigned z) {
  // x output row axis
  // y output column axis
  // z collapsing axis
  __shared__ double A[TILE_SIZE_DOUBLE][TILE_SIZE_DOUBLE];
  __shared__ double B[TILE_SIZE_DOUBLE][TILE_SIZE_DOUBLE];

  unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  double sum = 0.0;

  // loop over tiles of the reduction dim (z = K)
  for (int t = 0; t < (z + TILE_SIZE_DOUBLE - 1) / TILE_SIZE_DOUBLE; t++) {
    // load A tile
    if ((threadIdx.x + t * TILE_SIZE_DOUBLE) < z && idx_y < y)
      A[threadIdx.y][threadIdx.x] =
          a[(threadIdx.x + t * TILE_SIZE_DOUBLE) + idx_y * z];
    else
      A[threadIdx.y][threadIdx.x] = 0.0;

    // load B tile
    if (idx_x < x && (threadIdx.y + t * TILE_SIZE_DOUBLE) < z)
      B[threadIdx.y][threadIdx.x] =
          b[idx_x + (threadIdx.y + t * TILE_SIZE_DOUBLE) * x];
    else
      B[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads(); // make sure all threads have loaded their data

    for (unsigned i = 0; i < TILE_SIZE_DOUBLE; i++)
      sum += A[threadIdx.y][i] * B[i][threadIdx.x];

    __syncthreads(); // wait before overwriting
  }

  // write result
  if (idx_y < y && idx_x < x)
    c[idx_y * x + idx_x] = sum;
}

__global__ void gpu_kernel::matrixScalerMul(double *input, double scaler_value,
                                            double *output, unsigned x,
                                            unsigned y) {

  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx = id_x + id_y * x;

  if (id_x < x && id_y < y)
    output[lin_idx] = scaler_value * input[lin_idx];
}

__global__ void gpu_kernel::matrixAccuracyValue(double *confusion_matrix,
                                                double *accuracy, unsigned x,
                                                unsigned y) {
  unsigned i, true_values = 0;

  for (i = 0; i < x; i++)
    true_values += confusion_matrix[i + i * x];

  *accuracy = (double)true_values / y;
}

__global__ void gpu_kernel::matrixArgMax(double *a, double *b, unsigned x,
                                         unsigned y) {
  unsigned i, intr_y, index;
  double large = -1.0;

  intr_y = threadIdx.y + (blockIdx.y * blockDim.y);

  if (intr_y < y) {
    index = 0;
    for (i = 0; i < x; i++)
      if (large < a[i + intr_y * x]) {
        index = i;
        large = a[i + intr_y * x];
      }

    for (i = 0; i < x; i++)
      b[i + intr_y * x] = (index == i) ? 1 : 0;
  }
}

__global__ void gpu_kernel::matrixDotMul(double *input_A, double *input_B,
                                         double *input_C, double *output,
                                         unsigned x, unsigned y, unsigned z) {
  // m : features, n : neurons;
  // x : max feature, y : max neuron;
  unsigned intr_x, intr_y, intr_z, inp_lin, w_lin, res_lin;

  intr_x = threadIdx.x + (blockIdx.x * blockDim.x); // neuron axis
  intr_y = threadIdx.y + (blockIdx.y * blockDim.y); // batch axis
  intr_z = threadIdx.z + (blockIdx.z * blockDim.z); // batch axis

  inp_lin = intr_z + (intr_y * z); // input linear index.
  w_lin = intr_x + (intr_z * x);   // z = features + 1 for bias.
  res_lin =
      intr_x + (intr_y * x) + (intr_z * x * y); // resluting array input index.

  if (intr_x < x && intr_y < y && intr_z < z) {
    output[res_lin] = input_A[inp_lin] * input_B[w_lin]; // (double)
    // c[res_lin] = a[inp_lin];
  } else if (intr_x < x && intr_y < y && intr_z == z) {
    output[res_lin] = input_C[intr_x];
  }
}

__global__ void gpu_kernel::matrixDifferentialParameters(
    double *input, double *delta_output, double *difference,
    double *d_parameters, unsigned x, unsigned y, unsigned z) {
  unsigned indx_x, indx_y, indx_z, out_lin, inp_lin, diff_lin;

  indx_x = threadIdx.x + blockIdx.x * blockDim.x;
  indx_y = threadIdx.y + blockIdx.y * blockDim.y;
  indx_z = threadIdx.z + blockIdx.z * blockDim.z;

  if (indx_x < x && indx_y < y && indx_z < z) {
    out_lin = indx_x + indx_y * x + indx_z * x * y;
    inp_lin = indx_y + indx_z * y;
    diff_lin = indx_x + indx_z * x;

    d_parameters[out_lin] = delta_output[diff_lin] * difference[diff_lin];
    if (input)
      d_parameters[out_lin] *= input[inp_lin];
  }
}

__global__ void gpu_kernel::matrixDifferentialBiases(double *delta_output,
                                                     double *difference,
                                                     double *delta_biases,
                                                     unsigned x, unsigned y) {
  unsigned indx_x, indx_y, out_lin, diff_lin;

  indx_x = threadIdx.x + blockIdx.x * blockDim.x;
  indx_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (indx_x < x && indx_y < y) {
    out_lin = indx_x + indx_y * x;
    diff_lin = indx_x + indx_y * x;

    delta_biases[out_lin] =
        2 * delta_output[diff_lin] * difference[diff_lin]; //
  }
}

__global__ void
gpu_kernel::matrixDifferentialInput(double *weights, double *delta_output,
                                    double *difference, double *delta_input,
                                    unsigned x, unsigned y, unsigned z) {
  unsigned indx_x, indx_y, indx_z, out_lin, inp_lin, diff_lin;

  indx_x = threadIdx.x + blockIdx.x * blockDim.x;
  indx_y = threadIdx.y + blockIdx.y * blockDim.y;
  indx_z = threadIdx.z + blockIdx.z * blockDim.z;
  out_lin = indx_x + indx_y * x + indx_z * x * y;

  if (indx_x < x && indx_y < y & indx_z < z) {
    out_lin = indx_x + indx_y * x + indx_z * x * y;
    diff_lin = indx_y + indx_z * y;
    inp_lin = indx_x + indx_z * x;

    delta_input[out_lin] =
        2 * weights[inp_lin] * difference[diff_lin] * delta_output[diff_lin]; //
  }
}

__global__ void gpu_kernel::matrixRollingSum(double *input, double *output,
                                             unsigned x, unsigned y,
                                             unsigned z) {
  // input dimension: xyz, (z is adding axis, xy is bubble up axises).
  // output dimension: xy

  // x: neurons, y: features
  unsigned intr_x, intr_y, inp_lin, out_lin, i;
  double val;

  intr_x = threadIdx.x + (blockIdx.x * blockDim.x);
  intr_y = threadIdx.y + (blockIdx.y * blockDim.y);
  out_lin = intr_x + intr_y * x;

  if (intr_x < x && intr_y < y) {
    val = 0.0;
    for (i = 0; i < z; i++) {
      inp_lin = out_lin + i * x * y;
      val += input[inp_lin];
    }
    output[out_lin] = val;
  }
}

__global__ void gpu_kernel::tensorReduceSum(double *const input,
                                            double *const output,
                                            unsigned const vector_length,
                                            unsigned const num_vector) {
  unsigned idx_x, idx_y, lin_idx, shared_idx, shared_base;
  idx_x = threadIdx.x;
  idx_y = threadIdx.y + blockIdx.y + blockDim.y;

  lin_idx = idx_x + (idx_y * vector_length);
  shared_idx = threadIdx.x + threadIdx.y * blockDim.x;
  shared_base = threadIdx.y * blockDim.x;

  unsigned grouped_length = (vector_length / blockDim.x) * blockDim.x;
  unsigned remain = vector_length - grouped_length;

  // find rolling sum
  // reduction for elements out of range of powercle
  if (idx_y < num_vector) {
    global_shared_mem_d[shared_idx] = input[lin_idx];
    output[lin_idx] = global_shared_mem_d[shared_idx];
    if (idx_x < remain) {
      double sum = input[lin_idx + grouped_length];
      global_shared_mem_d[shared_idx] += sum;
      output[lin_idx + grouped_length] = sum;
    }
  }
  __syncthreads();

  // group reduction to get everything into shared memory
  if (idx_y < num_vector)
    for (unsigned i = blockDim.x; i < grouped_length; i += blockDim.x) {
      double sum = input[lin_idx + i];
      global_shared_mem_d[shared_idx] += sum;
      output[lin_idx + i] = sum;
    }
  __syncthreads();

  // rolling sum
  for (unsigned i = blockDim.x >> 1; i > 0; i >>= 1) {
    if (threadIdx.x < i && idx_y < num_vector)
      global_shared_mem_d[shared_idx] += global_shared_mem_d[shared_idx + i];
    __syncthreads();
  }

  // putting the result back to where it belong
  if (idx_y < num_vector) {
    output[lin_idx] = global_shared_mem_d[shared_base];
    if (idx_x < remain)
      output[lin_idx + grouped_length] /= global_shared_mem_d[shared_base];

    // group reduction to get everything into shared memory
    for (unsigned i = blockDim.x; i < grouped_length; i += blockDim.x)
      output[lin_idx + i] /= global_shared_mem_d[shared_base];
  }
}

__global__ void gpu_kernel::tensorReduceSumOffAxis(
    double *const input, double *const output, unsigned const vector_length,
    unsigned const inner_stride, unsigned const outer_stride) {
  unsigned idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned idx_y = threadIdx.y;
  unsigned idx_z = blockIdx.z;

  unsigned base_idx =
      idx_x + idx_y * inner_stride + idx_z * inner_stride * vector_length;
  unsigned shared_idx = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned shared_base = threadIdx.x;

  unsigned grouped_length = (vector_length / blockDim.y) * blockDim.y;
  unsigned remain = vector_length - grouped_length;

  // find rolling sum
  // reduction for elements out of range of powercle
  if (idx_x < inner_stride) {
    global_shared_mem_d[shared_idx] = input[base_idx];
    output[base_idx] = global_shared_mem_d[shared_idx];
    if (idx_y < remain) {
      double sum = input[base_idx + grouped_length * inner_stride];
      global_shared_mem_d[shared_idx] += sum;
      output[base_idx + grouped_length * inner_stride] = sum;
    }
  }
  __syncthreads();

  // group reduction to get everything into shared memory
  if (idx_x < inner_stride)
    for (unsigned i = blockDim.y; i < grouped_length; i += blockDim.y) {
      double sum = input[base_idx + i * inner_stride];
      global_shared_mem_d[shared_idx] += sum;
      output[base_idx + i * inner_stride] = sum;
    }
  __syncthreads();

  // rolling sum on shared memory
  for (unsigned i = blockDim.y >> 1; i > 0; i >>= 1) {
    if (threadIdx.y < i && idx_x < inner_stride)
      global_shared_mem_d[shared_idx] +=
          global_shared_mem_d[shared_idx + i * blockDim.x];
    __syncthreads();
  } // end of finding rolling sum

  // putting the result back to where it belong
  if (idx_x < inner_stride) {
    output[base_idx] /= global_shared_mem_d[shared_base];
    if (idx_y < remain)
      output[base_idx + grouped_length * inner_stride] /=
          global_shared_mem_d[shared_base];

    // group reduction to get everything into shared memory
    for (unsigned i = blockDim.y; i < grouped_length; i += blockDim.y)
      output[base_idx + i * inner_stride] /= global_shared_mem_d[shared_base];
  } // end of write
}

__global__ void gpu_kernel::matrixRelu(double *input_A, double *output, int x,
                                       int y) {
  // x: neuron, y: feature. m: max_neuron.
  unsigned id_x, id_y, id_z, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  id_z = blockIdx.z;

  if (id_x < x && id_y < y) {
    lin_idx = id_x + id_y * x + id_z * x * y;
    output[lin_idx] = input_A[lin_idx] * (input_A[lin_idx] > 0);
  }
}

__global__ void gpu_kernel::matrixSigmoid(double *a, double *output, int x,
                                          int y) {
  // x: neuron, y: feature. m: max_neuron.
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);

  if (id_x < x && id_y < y) {
    lin_idx = id_x + id_y * x;
    output[lin_idx] = 1.0f / (1 + exp(-1 * a[lin_idx]));
  }
}

__global__ void
gpu_kernel::matrixSub(double *const input_1, double *const input_2,
                      double *const output, const unsigned x_axis_dim,
                      const unsigned y_axis_dim, const unsigned plane_count) {
  unsigned idx_x, idx_y, idx_z;
  idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
  idx_y = threadIdx.y + (blockDim.y * blockIdx.y);
  idx_z = threadIdx.z + (blockDim.z * blockIdx.z);
  size_t index = idx_x + idx_y * x_axis_dim + idx_z * x_axis_dim * y_axis_dim;

  if (idx_x < x_axis_dim && idx_y < y_axis_dim && idx_z < plane_count)
    output[index] = input_1[index] - input_2[index];
}

__global__ void gpu_kernel::matrixSubBroadCast(
    double *const input_1, double *const input_2, double *const output,
    const unsigned nDimInput_1, const unsigned *const dimInput_1,
    const unsigned nDimInput_2, const unsigned *const dimInput_2,
    const unsigned plane_count, const unsigned plane_cout_b) {

  size_t idx_x = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t idx_y = threadIdx.y + (blockIdx.y * blockDim.y);
  size_t idx_z = threadIdx.z + (blockIdx.z * blockDim.z);

  __shared__ size_t x_axis_inp_1, y_axis_inp_1;
  __shared__ size_t x_axis_inp_2, y_axis_inp_2;
  __shared__ size_t a_plane_count, b_plane_count;
  __shared__ size_t idx_a, idx_b;
  __shared__ size_t i;

  __shared__ unsigned indices;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    x_axis_inp_1 = dimInput_1[0];
    y_axis_inp_1 = (nDimInput_1 > 1) ? dimInput_1[1] : 1;

    x_axis_inp_2 = dimInput_2[0];
    y_axis_inp_2 = (nDimInput_2 > 1) ? dimInput_2[1] : 1;

    a_plane_count = plane_count / dimInput_1[nDimInput_1 - 1];
    b_plane_count = plane_cout_b / dimInput_2[nDimInput_2 - 1];
    idx_a = idx_z;
    idx_b = 0;
    for (i = nDimInput_1 - 1; i > 1; i--) {
      indices = idx_a / a_plane_count;
      idx_a -= indices * a_plane_count;
      a_plane_count /= dimInput_1[i - 1];

      idx_b += indices * b_plane_count * (dimInput_2[i] != 1);
      b_plane_count /= dimInput_2[i - 1];
    }
    idx_b *= x_axis_inp_2 * y_axis_inp_2;
  }
  __syncthreads();

  size_t index_b = idx_b;

  size_t index =
      idx_x + idx_y * x_axis_inp_1 + idx_z * x_axis_inp_1 * y_axis_inp_1;
  index_b +=
      idx_x * (x_axis_inp_2 != 1) + idx_y * (y_axis_inp_2 != 1) * x_axis_inp_2;
  if (idx_x < x_axis_inp_1 && idx_y < y_axis_inp_1 && idx_z < plane_count) {
    output[index] = input_1[index] - input_2[index_b];
  }
}

__global__ void gpu_kernel::matrixLinear(double *a, double *d_a, int x, int y) {
  // x: neuron, y: batch.
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx = id_x + id_y * x;

  if (id_x < x && id_y < y) {
    d_a[lin_idx] = 1;
  }
}

__global__ void gpu_kernel::matrixSoftmax(double *a, double *softmax_sum,
                                          double *d_a, unsigned x, unsigned y) {
  unsigned i, id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx = id_x + (id_y * x);

  // Softmax calculation.
  if (id_x < x && id_y < y)
    d_a[lin_idx] = a[lin_idx] = exp(a[lin_idx]);
  __syncthreads();

  if (!id_x) {
    if (id_y < y) {
      softmax_sum[id_y] = 0;
      for (i = lin_idx; i < lin_idx + x; i++)
        softmax_sum[id_y] += a[i];
    }
  }
  __syncthreads();

  if (id_x < x && id_y < y)
    a[lin_idx] = a[lin_idx] / softmax_sum[id_y];
  __syncthreads();

  // Softmax derivative calculation.
  d_a[lin_idx] = 1;
  // if (id_x < x && id_y < y)
  // {
  //     d_a[lin_idx] = 0.0;
  //     for (i = 0; i < x; i++)
  //         d_a[lin_idx] +=a[lin_idx] * ((i == id_x) - a[i + id_y * x]);
  // }
}

__global__ void gpu_kernel::matrixSquaredError(double *a, double *b, unsigned x,
                                               unsigned y) {
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx = id_x + id_y * x;

  if (id_x < x && id_y < y)
    b[lin_idx] = pow((a[lin_idx]), 2);
}

__global__ void gpu_kernel::matrixSqrt(double *input_A, double *output,
                                       unsigned x, unsigned y) {
  // input dimension xy
  // output dimension xy
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx = id_x + id_y * x;

  if (id_x < x && id_y < y)
    output[lin_idx] = sqrt(input_A[lin_idx]);
}

__global__ void gpu_kernel::matrixFindMean(double *a, unsigned x, unsigned y,
                                           unsigned mean) {
  unsigned indx_x, indx_y, inp_lin;

  indx_x = threadIdx.x + blockIdx.x * blockDim.x;
  indx_y = threadIdx.y + blockIdx.y * blockDim.y;
  inp_lin = indx_x + indx_y * x;

  if (indx_x < x & indx_y < y)
    a[inp_lin] /= mean;
}

__global__ void gpu_kernel::matrixDifference(double *input_A, double *input_B,
                                             double *output_C, unsigned x,
                                             unsigned y) {
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);

  lin_idx = id_x + id_y * x;

  if (id_x < x && id_y < y) {
    output_C[lin_idx] =
        input_A[lin_idx] - input_B[lin_idx]; // a[lin_idx] - b[id_y];
  }
}

__global__ void gpu_kernel::matrixTranspose(double *input_A, double *output,
                                            unsigned x, unsigned y) {
  int idx_x, idx_y, inp_idx, out_idx;

  __shared__ double tile[TILE_SIZE_DOUBLE][TILE_SIZE_DOUBLE];

  idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  idx_y = blockDim.y * blockIdx.y + threadIdx.y;

  inp_idx = idx_x + idx_y * x;
  out_idx = idx_y + idx_x * y;

  if (idx_x < x && idx_y < y)
    tile[threadIdx.x][threadIdx.y] = input_A[inp_idx];

  __syncthreads();

  if (idx_x < x && idx_y < y)
    output[out_idx] = tile[threadIdx.x][threadIdx.y];
}