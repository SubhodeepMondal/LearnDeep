#ifdef CUDA_ENABLED
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "kernelmanager.h"
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opssoftmax::compute() {
  unsigned inner_stride = 1;
  for (unsigned i = 0; i < this->axis; i++)
    inner_stride *= this->inputs[0]->getDimensions()[i];

  unsigned softmax_axis_len = this->inputs[0]->getDimensions()[this->axis];
  unsigned outer_stride = 1;
  for (unsigned i = this->axis + 1; i < this->inputs[0]->getNoOfDimensions();
       i++)
    outer_stride *= this->inputs[0]->getDimensions()[i];

  std::float64_t *ptr[2];
  ptr[0] = this->inputs[0]->getData();
  ptr[1] = this->output->getData();
  kernel_dispatch(ptr, this->axis, softmax_axis_len, inner_stride,
                  outer_stride);
}

void Opssoftmax::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opssoftmax::initializeoutput(Tensor<std::float64_t> *outputs) {
  this->output = outputs;
  *(this->output) = *(inputs[0]);
}

void Opssoftmax::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opssoftmax::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

void Opssoftmax::kernel_dispatch(std::float64_t *const *const ptr,
                                 unsigned const axis, unsigned const axis_len,
                                 unsigned const inner_stride,
                                 unsigned const outer_stride) {
  KernelType kernel = get_global_kernel();
#ifdef CUDA_ENABLED
  switch (kernel) {
  case KernelType::GPU: {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_mat_softmax_f64(d_arr, axis, axis_len, inner_stride, outer_stride);
    break;
  }
  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_softmax_f64(ptr, axis, axis_len, inner_stride, outer_stride);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;
  case KernelType::CPU_SCALAR:
    cpu::__msoftmax(ptr, axis, axis_len, inner_stride, outer_stride);
    break;
  case KernelType::AUTO:
  default: {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_mat_softmax_f64(d_arr, axis, axis_len, inner_stride, outer_stride);
    break;
  }
  }
#else
  switch (kernel) {
  case KernelType::GPU:
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_softmax_f64(ptr, axis, axis_len, inner_stride, outer_stride);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;
  case KernelType::CPU_SCALAR:
    cpu::__msoftmax(ptr, axis, axis_len, inner_stride, outer_stride);
    break;
  case KernelType::AUTO:
  default:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_softmax_f64(ptr, axis, axis_len, inner_stride, outer_stride);
    } else {
      cpu::__msoftmax(ptr, axis, axis_len, inner_stride, outer_stride);
    }
    break;
  }
#endif
}
