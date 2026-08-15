// C++ Headers
#include <iostream>
#include <stdexcept>
#include <string>

// CUDA Headers
#include <cuda_runtime.h>

// Library Headers
#include "gpu_interface.cuh"
#include "gpu_micro_kernels.cuh"
#include <absl/log/log.h>
#include <core/LAS/gpu_interface.cuh>
#include <core/LAS/gpu_micro_kernels.cuh>

void gpu::gpu_mat_add_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0];
  unsigned y = arr[1];
  unsigned total_plane = 1;

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
  gpu_kernel::matrixSum<<<grid, block>>>(d_a, d_b, d_c, x, y, total_plane);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
};

void gpu::gpu_mat_add_broadcast_f64(double *const *const ptr,
                                    const unsigned nDimA, const unsigned *dimA,
                                    const unsigned nDimB, const unsigned *dimB,
                                    const bool isBoradCast) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];

  size_t dim_x_axis = 1;
  size_t dim_y_axis = 1;
  size_t total_plane = 1;
  size_t total_plane_b = 1;
  size_t nElemA = 1;
  size_t nElemB = 1;

  dim3 block;
  dim3 grid;

  if (nDimA > 2) {
    dim_x_axis = dimA[0];
    dim_y_axis = dimA[1];

    for (unsigned i = 2; i < nDimA; i++)
      total_plane *= dimA[i];
    nElemA = dimA[0] * dimA[1] * total_plane;

    for (unsigned i = 2; i < nDimB; i++)
      total_plane_b *= dimB[i];
    nElemB = dimB[0] * dimB[1] * total_plane_b;

    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    block.y = (32 > dimA[1]) ? dimA[1] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
    grid.y = (dimA[1] + block.y - 1) / block.y;
    grid.z = total_plane;
  } else if (nDimA > 1) {
    dim_x_axis = dimA[0];
    dim_y_axis = dimA[1];

    nElemA = dimA[0] * dimA[1];
    nElemB = dimB[0] * dimB[1];

    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    block.y = (32 > dimA[1]) ? dimA[1] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
    grid.y = (dimA[1] + block.y - 1) / block.y;
  } else {
    dim_x_axis = dimA[0];
    nElemA = dimA[0];
    nElemB = dimB[0];
    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
  }
  LOG(INFO) << "GPU kernel for matrix addition is running...";

  double *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, nElemA * sizeof(double));
  cudaMalloc((void **)&d_b, nElemB * sizeof(double));
  cudaMalloc((void **)&d_c, nElemA * sizeof(double));

  cudaMemcpy(d_a, a, nElemA * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, nElemB * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  if (!isBoradCast)
    gpu_kernel::matrixSum<<<grid, block>>>(d_a, d_b, d_c, dim_x_axis,
                                           dim_y_axis, total_plane);
  else {
    unsigned *dimA_device, *dimB_device;
    cudaMalloc((void **)&dimA_device, nDimA * sizeof(double));
    cudaMalloc((void **)&dimB_device, nDimB * sizeof(double));
    cudaMemcpy(dimA_device, dimA, nDimA * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dimB_device, dimB, nDimB * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    gpu_kernel::matrixSumBroadCast<<<grid, block>>>(
        d_a, d_b, d_c, nDimA, dimA_device, nDimB, dimB_device, total_plane,
        total_plane_b);
    cudaFree(dimA_device);
    cudaFree(dimB_device);
  }
  cudaMemcpy(c, d_c, nElemA * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
};

void gpu::gpu_mat_hadamard_mul_f64(double **ptr, const unsigned *dimA,
                                   unsigned nDimA) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = dimA[0];
  unsigned y = dimA[1];
  size_t total_plane = 1;

  for (unsigned i = 2; i < nDimA; i++)
    total_plane *= dimA[i];

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
  gpu_kernel::matrixHadamardMul<<<grid, block>>>(d_a, d_b, d_c, x, y,
                                                 total_plane);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
};

void gpu::gpu_mat_hadamard_mul_broadcast_f64(
    double *const *const ptr, const unsigned nDimA, const unsigned *dimA,
    const unsigned nDimB, const unsigned *dimB, const bool isBoradCast) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];

  size_t dim_x_axis = 1;
  size_t dim_y_axis = 1;
  size_t total_plane = 1;
  size_t total_plane_b = 1;
  size_t nElemA = 1;
  size_t nElemB = 1;

  dim3 block;
  dim3 grid;

  if (nDimA > 2) {
    dim_x_axis = dimA[0];
    dim_y_axis = dimA[1];

    for (unsigned i = 2; i < nDimA; i++)
      total_plane *= dimA[i];
    nElemA = dimA[0] * dimA[1] * total_plane;

    for (unsigned i = 2; i < nDimB; i++)
      total_plane_b *= dimB[i];
    nElemB = dimB[0] * dimB[1] * total_plane_b;

    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    block.y = (32 > dimA[1]) ? dimA[1] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
    grid.y = (dimA[1] + block.y - 1) / block.y;
    grid.z = total_plane;
  } else if (nDimA > 1) {
    dim_x_axis = dimA[0];
    dim_y_axis = dimA[1];

    nElemA = dimA[0] * dimA[1];
    nElemB = dimB[0] * dimB[1];

    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    block.y = (32 > dimA[1]) ? dimA[1] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
    grid.y = (dimA[1] + block.y - 1) / block.y;
  } else {
    dim_x_axis = dimA[0];
    nElemA = dimA[0];
    nElemB = dimB[0];
    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
  }
  LOG(INFO) << "GPU kernel for matrix addition is running...";

  double *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, nElemA * sizeof(double));
  cudaMalloc((void **)&d_b, nElemB * sizeof(double));
  cudaMalloc((void **)&d_c, nElemA * sizeof(double));

  cudaMemcpy(d_a, a, nElemA * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, nElemB * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  if (!isBoradCast)
    gpu_kernel::matrixHadamardMul<<<grid, block>>>(d_a, d_b, d_c, dim_x_axis,
                                                   dim_y_axis, total_plane);
  else {
    unsigned *dimA_device, *dimB_device;
    cudaMalloc((void **)&dimA_device, nDimA * sizeof(double));
    cudaMalloc((void **)&dimB_device, nDimB * sizeof(double));
    cudaMemcpy(dimA_device, dimA, nDimA * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dimB_device, dimB, nDimB * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    gpu_kernel::matrixMulBroadCast<<<grid, block>>>(
        d_a, d_b, d_c, nDimA, dimA_device, nDimB, dimB_device, total_plane,
        total_plane_b);
    cudaFree(dimA_device);
    cudaFree(dimB_device);
  }
  cudaMemcpy(c, d_c, nElemA * sizeof(double), cudaMemcpyDeviceToHost);
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
  gpu_kernel::matrixResuffledMul<<<grid, block>>>(d_a, d_b, d_c, x, y, z);
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

void gpu::gpu_mat_sub_broadcast_f64(double *const *const ptr,
                                    const unsigned nDimA, const unsigned *dimA,
                                    const unsigned nDimB, const unsigned *dimB,
                                    const bool isBoradCast) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];

  size_t dim_x_axis = 1;
  size_t dim_y_axis = 1;
  size_t total_plane = 1;
  size_t total_plane_b = 1;
  size_t nElemA = 1;
  size_t nElemB = 1;

  dim3 block;
  dim3 grid;

  if (nDimA > 2) {
    dim_x_axis = dimA[0];
    dim_y_axis = dimA[1];

    for (unsigned i = 2; i < nDimA; i++)
      total_plane *= dimA[i];
    nElemA = dimA[0] * dimA[1] * total_plane;

    for (unsigned i = 2; i < nDimB; i++)
      total_plane_b *= dimB[i];
    nElemB = dimB[0] * dimB[1] * total_plane_b;

    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    block.y = (32 > dimA[1]) ? dimA[1] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
    grid.y = (dimA[1] + block.y - 1) / block.y;
    grid.z = total_plane;
  } else if (nDimA > 1) {
    dim_x_axis = dimA[0];
    dim_y_axis = dimA[1];

    nElemA = dimA[0] * dimA[1];
    nElemB = dimB[0] * dimB[1];

    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    block.y = (32 > dimA[1]) ? dimA[1] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
    grid.y = (dimA[1] + block.y - 1) / block.y;
  } else {
    dim_x_axis = dimA[0];
    nElemA = dimA[0];
    nElemB = dimB[0];
    block.x = (32 > dimA[0]) ? dimA[0] : 32;
    grid.x = (dimA[0] + block.x - 1) / block.x;
  }
  LOG(INFO) << "GPU kernel for matrix addition is running...";

  double *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, nElemA * sizeof(double));
  cudaMalloc((void **)&d_b, nElemB * sizeof(double));
  cudaMalloc((void **)&d_c, nElemA * sizeof(double));

  cudaMemcpy(d_a, a, nElemA * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, nElemB * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  if (!isBoradCast)
    gpu_kernel::matrixSub<<<grid, block>>>(d_a, d_b, d_c, dim_x_axis,
                                           dim_y_axis, total_plane);
  else {
    unsigned *dimA_device, *dimB_device;
    cudaMalloc((void **)&dimA_device, nDimA * sizeof(double));
    cudaMalloc((void **)&dimB_device, nDimB * sizeof(double));
    cudaMemcpy(dimA_device, dimA, nDimA * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dimB_device, dimB, nDimB * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    gpu_kernel::matrixSubBroadCast<<<grid, block>>>(
        d_a, d_b, d_c, nDimA, dimA_device, nDimB, dimB_device, total_plane,
        total_plane_b);
    cudaFree(dimA_device);
    cudaFree(dimB_device);
  }
  cudaMemcpy(c, d_c, nElemA * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
};

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

void gpu::gpu_reduce_sum_f64(double *const *const ptr,
                             unsigned const *const arr) {
  double *input = ptr[0];
  double *output = ptr[1];

  LOG(INFO) << "GPU kernel for reduction sum is running...";

  unsigned no_of_dims = arr[0];
  // size_t total_plane(1);
  unsigned axis = arr[no_of_dims + 1];
  size_t vector_length = arr[axis + 1];
  size_t inner_stride(1);
  size_t outer_stride(1);

  for (unsigned i = 1; i <= axis; i++)
    inner_stride *= arr[i];

  for (unsigned i = axis + 2; i <= no_of_dims; i++)
    outer_stride *= arr[i];

  // size_t no_of_line = inner_stride * outer_stride;
  unsigned power_of_2;
  dim3 grid, block;
  // size_t req_shared_mem = 1024;
  double *d_a = nullptr, *d_c = nullptr;
  unsigned num_vector = inner_stride * outer_stride;
  const size_t total_elements = vector_length * num_vector;

  auto check_cuda = [](cudaError_t err, const char *operation) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string(operation) +
                               " failed: " + cudaGetErrorString(err));
    }
  };

  check_cuda(cudaMalloc((void **)&d_a, total_elements * sizeof(double)),
             "cudaMalloc reducesum input");
  check_cuda(cudaMalloc((void **)&d_c, total_elements * sizeof(double)),
             "cudaMalloc reduce sum output");

  check_cuda(cudaMemcpy(d_a, input, total_elements * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy reducesum host-to-device");
  if (axis) {
    if (outer_stride > (1e16 - 1))
      LOG(ERROR) << "Kernel capasity exceeded!";
    power_of_2 =
        vector_length > 1 ? 1u << (30 - __builtin_clz(vector_length)) : 1;
    block.y = fmin(power_of_2, 1024);
    block.x = 1024 / block.y;
    grid.x = (inner_stride + block.x - 1) / block.x;
    grid.z = outer_stride;
    gpu_kernel::tensorReduceSumOffAxis<<<grid, block>>>(
        d_a, d_c, vector_length, inner_stride, outer_stride);
    check_cuda(cudaGetLastError(), "tensorSoftmaxOffAxis launch");
  } else {
    power_of_2 =
        vector_length > 1 ? 1u << (30 - __builtin_clz(vector_length)) : 1;
    block.x = fmin(power_of_2, 1024);
    block.y = (num_vector > (1024 / block.x)) ? 1024 / block.x : num_vector;
    grid.y = (num_vector + block.y - 1) / block.y;
    gpu_kernel::tensorReduceSum<<<grid, block>>>(d_a, d_c, vector_length,
                                                 num_vector);
    check_cuda(cudaGetLastError(), "tensor Reduction Sum launch");
  }
  check_cuda(cudaDeviceSynchronize(), "reduce sum kernel execution");
  check_cuda(cudaMemcpy(output, d_c, total_elements * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy reduction sum device-to-host");
  cudaFree(d_a);
  cudaFree(d_c);
}

void gpu::gpu_mat_relu_f64(double **ptr, const unsigned nDim,
                           unsigned int const *arr) {

  double *a = ptr[0];
  double *c = ptr[1];
  unsigned x = arr[0];
  unsigned y = arr[1];

  size_t total_plane = 1;
  if (nDim > 2)
    for (unsigned i = 2; i < nDim; i++)
      total_plane *= arr[i];
  LOG(INFO) << "GPU kernel for matrix ReLU is running...";

  dim3 block;
  dim3 grid;
  block.x = (32 > x) ? x : 32;
  block.y = (32 > y) ? y : 32;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;
  grid.z = total_plane;

  double *d_a, *d_c;

  cudaMalloc((void **)&d_a, x * y * total_plane * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * total_plane * sizeof(double));

  cudaMemcpy(d_a, a, x * y * total_plane * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixRelu<<<grid, block>>>(d_a, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * total_plane * sizeof(double),
             cudaMemcpyDeviceToHost);
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

/**
 * @brief gpu_softmax_kernel: takes float64 bit tensor and produces softmax on
 * given axis
 * @param ptr double pointer receives base pointer of input & output
 * @param arr encoded value received vector_size and number of vectors & axis
 * along which softmax to be performed */
void gpu::gpu_mat_softmax_f64(double *const *const ptr, unsigned axis,
                              unsigned const vector_length,
                              unsigned const inner_stride,
                              unsigned const outer_stride) {

  double *a = ptr[0];
  double *c = ptr[1];

  LOG(INFO) << "GPU kernel for matrix Softmax is running...";
  unsigned power_of_2;
  dim3 grid, block;
  size_t req_shared_mem = 1024;
  double *d_a = nullptr, *d_c = nullptr;
  unsigned num_vector = inner_stride * outer_stride;
  const size_t total_elements = vector_length * num_vector;

  auto check_cuda = [](cudaError_t err, const char *operation) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string(operation) + " failed: " +
                               cudaGetErrorString(err));
    }
  };

  check_cuda(cudaMalloc((void **)&d_a, total_elements * sizeof(double)),
             "cudaMalloc softmax input");
  check_cuda(cudaMalloc((void **)&d_c, total_elements * sizeof(double)),
             "cudaMalloc softmax output");

  check_cuda(cudaMemcpy(d_a, a, total_elements * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy softmax host-to-device");
  if (axis) {
    if (outer_stride > (1e16 - 1))
      LOG(ERROR) << "Kernel capasity exceeded!";
    power_of_2 =
        vector_length > 1 ? 1u << (30 - __builtin_clz(vector_length)) : 1;
    block.y = fmin(power_of_2, 1024);
    block.x = 1024 / block.y;
    grid.x = (inner_stride + block.x - 1) / block.x;
    grid.z = outer_stride;
    gpu_kernel::
        tensorSoftmaxOffAxis<<<grid, block, req_shared_mem * sizeof(double)>>>(
            d_a, d_c, vector_length, inner_stride, outer_stride);
    check_cuda(cudaGetLastError(), "tensorSoftmaxOffAxis launch");
  } else {
    power_of_2 =
        vector_length > 1 ? 1u << (30 - __builtin_clz(vector_length)) : 1;
    block.x = fmin(power_of_2, 1024);
    block.y = 1024 / block.x;
    grid.y = (num_vector + block.y - 1) / block.y;
    gpu_kernel::tensorSoftmax<<<grid, block, req_shared_mem * sizeof(double)>>>(
        d_a, d_c, vector_length, num_vector);
    check_cuda(cudaGetLastError(), "tensorSoftmax launch");
  }
  check_cuda(cudaDeviceSynchronize(), "softmax kernel execution");
  check_cuda(cudaMemcpy(c, d_c, total_elements * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy softmax device-to-host");
  cudaFree(d_a);
  cudaFree(d_c);
}

void gpu::gpu_mat_transpose_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *c = ptr[1];
  unsigned x = arr[0];
  unsigned y = arr[1];

  LOG(INFO) << "GPU kernel for matrix Transpose is running...";

  dim3 block;
  dim3 grid;
  block.x = (16 > x) ? x : 16;
  block.y = (16 > y) ? y : 16;
  grid.x = (x + block.x - 1) / block.x;
  grid.y = (y + block.y - 1) / block.y;

  double *d_a, *d_c;

  cudaMalloc((void **)&d_a, x * y * sizeof(double));
  cudaMalloc((void **)&d_c, x * y * sizeof(double));

  cudaMemcpy(d_a, a, x * y * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err;
  gpu_kernel::matrixTranspose<<<grid, block>>>(d_a, d_c, x, y);
  cudaMemcpy(c, d_c, x * y * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_c);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err);
  }
}
