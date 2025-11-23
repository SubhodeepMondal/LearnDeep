#include "gpu_interface.cuh"
#include "gpu_micro_kernels.cuh"
#include <LAS/gpu_interface.cuh>
#include <LAS/gpu_micro_kernels.cuh>
#include <absl/log/log.h>
#include <cuda_runtime.h>
#include <iostream>

void gpu::gpu_mat_add_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix addition is running...";

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_b, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixSum<<<grid, block>>>(d_a, d_b, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
};

void gpu::gpu_mat_hadamard_mul_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix element wise multipliction is running...";

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_b, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixHadamardMul<<<grid, block>>>(d_a, d_b, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
};

void gpu::gpu_mat_mul_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0]; // output row size
  unsigned y = arr[2]; // output column size
  unsigned z = arr[1]; // collapsing axis

  LOG(INFO) << "GPU kernel for matrix element wise multipliction is running...";

  dim3 block;
  dim3 grid;
  block.x = (16 > x) ? x : 16;
  block.y = (16 > z) ? z : 16;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, y * z * sizeof(double));
  cudaMalloc((void **)&d_b, x * z * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, y * z * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, x * z * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixTiledMul<<<grid, block>>>(d_a, d_b, d_c, x, y, z);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}

void gpu::gpu_mat_scale_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double scaling_factor = ptr[1][0];
  double *c = ptr[2];

  unsigned x = arr[0]; // x output row size
  unsigned y = arr[1]; // y k row

  LOG(INFO) << "GPU kernel for matrix scaling is running...";
  dim3 block;
  dim3 grid;

  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_c;
  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);

  cudaError_t err;
  gpu_kernel::matrixScalerMul<<<grid, block>>>(d_a, scaling_factor, d_c, x, y);

  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_c);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}

void gpu::gpu_mat_sub_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix addition is running...";

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_b, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixSub<<<grid, block>>>(d_a, d_b, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}

void gpu::gpu_mat_sqrt_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *c = ptr[1];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix square root is running...";

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixSqrt<<<grid, block>>>(d_a, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}

void gpu::gpu_mat_relu_f64(double **ptr, unsigned int *arr) {

  double *a = ptr[0];
  double *c = ptr[1];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix ReLU is running...";

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixRelu<<<grid, block>>>(d_a, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}

void gpu::gpu_mat_sigmoid_f64(double **ptr, unsigned int *arr) {

  double *a = ptr[0];
  double *c = ptr[1];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix Sigmoid is running...";

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixSigmoid<<<grid, block>>>(d_a, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}

void gpu::gpu_mat_softmax_f64(double **ptr, unsigned int *arr) {

  double *a = ptr[0];
  double *c = ptr[1];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix Softmax is running...";

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_c, *d_softmax_sum;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));
  cudaMalloc((void **)&d_softmax_sum, x * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixSoftmax<<<grid, block>>>(d_a, d_softmax_sum, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_c);
  cudaFree(d_softmax_sum);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}

void gpu::gpu_mat_transpose_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *c = ptr[1];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix Transpose is running...";

  dim3 block;
  dim3 grid;
  block.x = TILE_SIZE_DOUBLE;
  block.y = TILE_SIZE_DOUBLE;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixTiledTranspose<<<grid, block>>>(d_a, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}