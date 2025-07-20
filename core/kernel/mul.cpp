#ifdef CUDA_ENABLED
#include <LAS/gpu_interface.cuh>
#endif

#include <LAS/CPULibrary.h>
#include <LAS/avx2_micro_kernels.h>
#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>

void Opsmul::recursive_iterator(unsigned index, unsigned *dimension_arr,
                                std::string function_name, unsigned *ui_arr,
                                std::float64_t *dl_arr,
                                Tensor<std::float64_t> *misc_arr) {
  if (index < 2) {
    unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
    unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
        c_index;

    inpA_x = (inputs[0]->getNoOfDimensions() > 0)
                 ? inputs[0]->getDimensions()[0]
                 : 1;
    inpA_y = (inputs[0]->getNoOfDimensions() > 1)
                 ? inputs[0]->getDimensions()[1]
                 : 1;

    inpB_x = (inputs[1]->getNoOfDimensions() > 0)
                 ? inputs[1]->getDimensions()[0]
                 : 1;
    inpB_y = (inputs[1]->getNoOfDimensions() > 1)
                 ? inputs[1]->getDimensions()[1]
                 : 1;

    out_x = (output->getNoOfDimensions() > 0) ? output->getDimensions()[0] : 1;
    out_y = (output->getNoOfDimensions() > 1) ? output->getDimensions()[1] : 1;

    a_plane_size = inpA_x * inpA_y;
    b_plane_size = inpB_x * inpB_y;
    c_plane_size = out_x * out_y;

    a_index = b_index = c_index = 0;
    if (inputs[1]->getNoOfDimensions() > 2)
      for (i = 2; i < inputs[1]->getNoOfDimensions(); i++) {
        a_index += a_plane_size * dimension_arr[i];
        b_index += b_plane_size * dimension_arr[i];
        c_index += c_plane_size * dimension_arr[i];

        a_plane_size *= inputs[0]->getDimensions()[i];
        b_plane_size *= inputs[1]->getDimensions()[i];
        c_plane_size *= output->getDimensions()[i];
      }

    /* code */
    unsigned a[2];
    std::float64_t *ptr[3];

    a[0] = inpA_x;
    a[1] = inpA_y;

    ptr[0] = inputs[0]->getData() + a_index;
    ptr[1] = inputs[1]->getData() + b_index;
    ptr[2] = output->getData() + c_index;
    kernel_dispatch(ptr, a);

  } else {
    for (unsigned i = 0; i < inputs[0]->getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, function_name, NULL, NULL,
                         NULL);
    }
  }
}

void Opsmul::compute() {
  unsigned dim_x, dim_y;
  dim_x = inputs[0]->getDimensions()[0];
  dim_y = inputs[1]->getDimensions()[1];

  unsigned *arr = new unsigned[inputs[0]->getNoOfDimensions()];

  recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr,
                     "matrix_element_wise_multiplication", NULL, NULL, NULL);

  delete[] arr;
}

void Opsmul::initilizeinputs(Tensor<std::float64_t> **inputs,
                             unsigned no_of_inputs) {
  unsigned i;
  this->no_of_inputs = no_of_inputs;

  this->inputs = new Tensor<std::float64_t> *[this->no_of_inputs];

  for (i = 0; i < this->no_of_inputs; i++) {
    this->inputs[i] = inputs[i];
  }
}

void Opsmul::initilizeoutput(Tensor<std::float64_t> *output) {
  this->output = output;

  *(this->output) = *(inputs[0]);
}

void Opsmul::printinputs() {
  unsigned i;
  for (i = 0; i < this->no_of_inputs; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsmul::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

void Opsmul::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {

#ifdef CUDA_ENABLED
  double *d_arr[3];
  d_arr[0] = reinterpret_cast<double *>(ptr[0]);
  d_arr[1] = reinterpret_cast<double *>(ptr[1]);
  d_arr[2] = reinterpret_cast<double *>(ptr[2]);

  gpu::gpu_mat_mul_f64(d_arr, arr);
#else
  if (__builtin_cpu_supports("avx2")) {
    avx2::avx2_mul_f64(ptr, arr);
  } else {
    cpu::__melementwisemul(ptr, arr);
  }
#endif
}