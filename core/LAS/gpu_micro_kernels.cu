#include "gpu_micro_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE_DOUBLE 16

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

__global__ void gpu_kernel::matrixSum(double *a, double *b, double *c,
                                      unsigned x, unsigned y) {
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx = id_x + id_y * x;

  if (id_x < x && id_y < y)
    c[lin_idx] = a[lin_idx] + b[lin_idx];
}

__global__ void gpu_kernel::matrixHadamardMul(double *a, double *b, double *c,
                                              unsigned x, unsigned y) {
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx = id_x + id_y * x;

  if (id_x < x && id_y < y)
    c[lin_idx] = a[lin_idx] * b[lin_idx];
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

__global__ void gpu_kernel::matrixRelu(double *input_A, double *output, int x,
                                       int y) {
  // x: neuron, y: feature. m: max_neuron.
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);

  if (id_x < x && id_y < y) {
    lin_idx = id_x + id_y * x;
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

__global__ void gpu_kernel::matrixSub(double *input_A, double *input_B,
                                      double *output, unsigned x, unsigned y) {
  unsigned id_x, id_y, lin_idx;
  id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  lin_idx = id_x + id_y * x;

  if (id_x < x && id_y < y)
    output[lin_idx] = input_A[lin_idx] - input_B[lin_idx];
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