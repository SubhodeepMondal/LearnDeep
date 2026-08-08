#ifdef CUDA_ENABLED
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opssoftmax::compute() {
  arr = new unsigned[this->inputs[0]->getNoOfDimensions() + 2];
  std::float64_t *ptr[2];
  ptr[0] = this->inputs[0]->getData();
  ptr[1] = this->output->getData();

  arr[0] = this->axis;
  arr[1] = this->inputs[0]->getNoOfDimensions();

  for (unsigned i = 0; i < this->inputs[0]->getNoOfDimensions(); i++)
    arr[i + 2] = this->inputs[0]->getDimensions()[i];

  kernel_dispatch(ptr, arr);
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
                                 unsigned *arr) {
#ifdef CUDA_ENABLED
  double *d_arr[3];
  d_arr[0] = reinterpret_cast<double *>(ptr[0]);
  d_arr[1] = reinterpret_cast<double *>(ptr[1]);
  d_arr[2] = reinterpret_cast<double *>(ptr[2]);

  gpu::gpu_mat_softmax_f64(d_arr, arr);
#else
  if (__builtin_cpu_supports("avx2")) {
    // avx2::avx2_softmax_f64(ptr, arr);
    cpu::__msoftmax(ptr, arr);
  } else {
    cpu::__msoftmax(ptr, arr);
  }
#endif
}
