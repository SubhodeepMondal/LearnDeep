#include "opskernel.h"
#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "kernelmanager.h"
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opslog::compute() {
  unsigned *arr = new unsigned[this->inputs[0]->getNoOfDimensions() + 1];

  std::float64_t *ptr[2];

  ptr[0] = inputs[0]->getData();
  ptr[1] = output->getData();

  arr[0] = this->inputs[0]->getNoOfDimensions();
  for (unsigned i = 0; i < arr[0]; i++)
    arr[i + 1] = this->inputs[0]->getDimensions()[i];

  kernel_dispatch(ptr, arr);
  delete[] arr;
}

void Opslog::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opslog::initializeoutput(Tensor<std::float64_t> *outputs) {
  this->output = outputs;
  // *(this->output) = *(inputs[0]);
  this->output->reshape(this->inputs[0]->getNoOfDimensions(),
                        this->inputs[0]->getDimensions());
}

void Opslog::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opslog::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

void Opslog::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {
  KernelType kernel = get_global_kernel();
#ifdef ENABLE_CUDA
  switch (kernel) {
  case KernelType::GPU: {
    double *d_arr[3];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    d_arr[2] = reinterpret_cast<double *>(ptr[2]);
    gpu::gpu_log_f64(d_arr, arr);
    break;
  }
  case KernelType::AVX2:
    if (!this->warning_handled) {
      std::cerr << "AVX2 kernel requested but, log has no AVX2 "
                   "implementation, falling back to CPU kernel\n";
      this->warning_handled = true;
    }
  case KernelType::CPU_SCALAR:
  case KernelType::AUTO:
  default:
    cpu::__mlog(ptr, arr);
    break;
  }
#else
  switch (kernel) {
  case KernelType::GPU:
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
  case KernelType::AVX2:
    if (!this->warning_handled) {
      std::cerr << "AVX2 kernel requested but, log has no AVX2 "
                   "implementation, falling back to CPU kernel\n";
      this->warning_handled = true;
    }
  case KernelType::CPU_SCALAR:
    cpu::__mlog(ptr, arr);
    break;
  case KernelType::AUTO:
  default:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_scale_f64(ptr, arr);
    } else {
      cpu::__mlog(ptr, arr);
    }
    break;
  }
#endif
}
