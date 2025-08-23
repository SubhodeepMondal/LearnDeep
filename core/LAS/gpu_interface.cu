#include <LAS/gpu_interface.cuh>
#include <LAS/gpu_micro_kernels.cuh>
#include <cuda_runtime.h>
#include <iostream>

void gpu::gpu_mat_add_f64(double **ptr, unsigned *arr)
{

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0];
  unsigned y = arr[1];

  std::cout << "GPU kernel for matrix addition is running..." << std::endl;

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
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
};

void gpu::gpu_mat_hadamard_mul_f64(double **ptr, unsigned *arr)
{

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0];
  unsigned y = arr[1];

  std::cout << "GPU kernel for matrix element wise multipliction is running..." << std::endl;

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
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
};

void gpu::gpu_mat_mul_f64(double **ptr, unsigned *arr)
{
  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0]; // x output row size
  unsigned y = arr[1]; // y k row
  unsigned z = arr[2]; // z output column size

  std::cout << "GPU kernel for matrix element wise multipliction is running..." << std::endl;

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > z) ? z : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (z + block.y - 1) / block.y;

  double *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_b, y * z * sizeof(double));
  cudaMalloc((void **)&d_c, x * z * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixMul<<<grid, block>>>(d_a, d_b, d_c, x, y, z);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
};