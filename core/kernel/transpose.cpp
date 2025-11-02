#ifdef CUDA_ENABLED
#include <LAS/gpu_interface.cuh>
#endif

#include <LAS/CPULibrary.h>
#include <LAS/avx2_micro_kernels.h>

#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>

void Opstranspose::recursive_iterator(unsigned index, unsigned *dimension_arr,
                                      std::string function_name,
                                      unsigned *ui_arr, std::float64_t *dl_arr,
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

    out_x = (output->getNoOfDimensions() > 0) ? output->getDimensions()[0] : 1;
    out_y = (output->getNoOfDimensions() > 1) ? output->getDimensions()[1] : 1;

    a_plane_size = inpA_x * inpA_y;
    c_plane_size = out_x * out_y;

    a_index = b_index = c_index = 0;
    if (inputs[0]->getNoOfDimensions() > 2)
      for (i = 2; i < inputs[0]->getNoOfDimensions(); i++) {
        a_index += a_plane_size * dimension_arr[i];
        c_index += c_plane_size * dimension_arr[i];

        a_plane_size *= inputs[0]->getDimensions()[i];
        c_plane_size *= output->getDimensions()[i];
      }
    unsigned a[2];
    std::float64_t *ptr[3];

    a[0] = inpA_x;
    a[1] = inpA_y;

    ptr[0] = inputs[0]->getData() + a_index;
    // ptr[1] = dl_arr;
    ptr[1] = output->getData() + c_index;

    kernel_dispatch(ptr, a);
  } else {
    for (unsigned i = 0; i < inputs[0]->getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, function_name, ui_arr,
                         dl_arr, misc_arr);
    }
  }
};

void Opstranspose::compute() {

  unsigned *arr = new unsigned[inputs[0]->getNoOfDimensions()];

  recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, "matrix_addition",
                     NULL, NULL, NULL);

  delete[] arr;
}

void Opstranspose::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->no_of_inputs = 1;
  this->inputs.push_back(inputs[0]);
}

void Opstranspose::initializeoutput(Tensor<std::float64_t> *output) {
  this->output = output;

  std::vector<unsigned> dims(inputs[0]->getDimensions(),
                             inputs[0]->getDimensions() +
                                 inputs[0]->getNoOfDimensions());
  // Swap dimensions
  dims[0] = dims[0] + dims[1];
  dims[1] = dims[0] - dims[1];
  dims[0] = dims[0] - dims[1];

  this->output->reshape(dims.size(), dims.data());
}

void Opstranspose::printinputs() {
  unsigned i;
  for (i = 0; i < this->no_of_inputs; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opstranspose::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *Opstranspose::getOutgoingGradientTensor(
    Tensor<std::float64_t> *gradient_input) {
  int i, it;
  bool flag = false;
  for (i = 0; i < 2; i++)
    if (this->inputs[i] == gradient_input) {
      it = i;
      flag = true;
      break;
    }

  if (flag) {
    // LOG(INFO) << "Requested gradint for the tensor found.\n";
    return this->outgoing_gradient;
  } else {
    // LOG(FATAL) << "Requested gradint for the tensor doesn't exist.\n";
    return NULL;
  }
}

// Tensor<std::float64_t> *
// Opstranspose::getIncomingGradientTensor(Tensor<std::float64_t> *tensor) {
//   return incoming_gradient;
// }

void Opstranspose::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {

  // #ifdef CUDA_ENABLED
  //   double *d_arr[3];
  //   d_arr[0] = reinterpret_cast<double *>(ptr[0]);
  //   d_arr[1] = reinterpret_cast<double *>(ptr[1]);
  //   d_arr[2] = reinterpret_cast<double *>(ptr[2]);

  //   gpu::gpu_mat_add_f64(d_arr, arr);
  // #else
  //   if (__builtin_cpu_supports("avx2")) {
  //     avx2::avx2_add_f64(ptr, arr);
  //   } else {
  cpu::__mtranspose(ptr, arr);
  //   }
  // #endif
}
