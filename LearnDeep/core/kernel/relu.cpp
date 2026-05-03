#ifdef CUDA_ENABLED
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "kernelmanager.h"
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opsrelu::compute() {
  std::float64_t *ptr[2];
  ptr[0] = this->inputs[0]->getData();
  ptr[1] = this->output->getData();

  this->kernel_dispatch(ptr, this->inputs[0]->getNoOfDimensions(),
                        this->inputs[0]->getDimensions());
}

void Opsrelu::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opsrelu::initializeoutput(Tensor<std::float64_t> *outputs) {
  this->output = outputs;
  *(this->output) = *(inputs[0]);
}

void Opsrelu::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsrelu::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

void Opsrelu::kernel_dispatch(std::float64_t **ptr, const unsigned nDim,
                              unsigned const *arr) {

  KernelType kernel = get_global_kernel();
#ifdef CUDA_ENABLED
  bool gpu_available = true;
#else
  bool gpu_available = false;
#endif

  switch (kernel) {

  case KernelType::GPU:
#ifdef CUDA_ENABLED
  {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_mat_relu_f64(d_arr, nDim, arr);
  }
#else
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
#endif
  break;

  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_relu_f64(ptr, nDim, arr);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;

  case KernelType::CPU_SCALAR:
    cpu::__mrelu(ptr, arr);
    break;

  case KernelType::AUTO:
  default:
#ifdef CUDA_ENABLED
  {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_mat_relu_f64(d_arr, nDim, arr);
  }
#else
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_relu_f64(ptr, nDim, arr);
    } else {
      cpu::__mrelu(ptr, arr);
    }
#endif
  break;
  }
}
