#include <LAS/gpu_interface.h>
#include <LAS/gpu_micro_kernels.cuh>
#include <cuda_runtime.h>



void gpu::gpu_mat_add_f64(double **ptr, unsigned *arr) {

  double *a = ptr[0];
  double *b = ptr[1];
  double *c = ptr[2];
  unsigned x = arr[0];
  unsigned y = arr[1];

  gpu::matrixSum<<<1,1>>>(a, b, c, x, y); 
}