#include "../framework/MathLibrary.h"
#include "opskernel.h"

void Opsscale::recursive_iterator(unsigned index, unsigned *dimension_arr,
                                  tensor<double> input_a,
                                  tensor<double> *input_b,
                                  tensor<double> &output,
                                  std::string function_name, unsigned *ui_arr,
                                  double *dl_arr, tensor<double> *misc_arr) {
  if (index < 2) {
    unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
    unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
        c_index;

    inpA_x = (input_a.getNoOfDimensions() > 0) ? input_a.getDimensions()[0] : 1;
    inpA_y = (input_a.getNoOfDimensions() > 1) ? input_a.getDimensions()[1] : 1;

    out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
    out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

    a_plane_size = inpA_x * inpA_y;
    c_plane_size = out_x * out_y;

    a_index = b_index = c_index = 0;
    if (input_a.getNoOfDimensions() > 2)
      for (i = 2; i < input_a.getNoOfDimensions(); i++) {
        a_index += a_plane_size * dimension_arr[i];
        c_index += c_plane_size * dimension_arr[i];

        a_plane_size *= input_a.getDimensions()[i];
        c_plane_size *= output.getDimensions()[i];
      }
    unsigned a[2];
    double *ptr[3];

    a[0] = inpA_x;
    a[1] = inpA_y;

    ptr[0] = input_a.getData() + a_index;
    ptr[1] = dl_arr;
    ptr[2] = output.getData() + c_index;

    cpu::__mscalermul(ptr, a);
  } else {
    // std::cout << "inside else\n";
    for (unsigned i = 0; i < input_a.getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, input_a, input_b, output,
                         function_name, ui_arr, dl_arr, misc_arr);
    }
  }
};

void Opsscale::compute() {
  unsigned *arr;

  arr = new unsigned[inputs[0]->getNoOfDimensions()];

  recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, *(inputs[0]),
                     NULL, *(output), "matrix_scaler_multiplication", NULL,
                     &scale_factor, NULL);
  delete[] arr;
}

void Opsscale::initilizeinputs(tensor<double> **inputs, double scale_factor) {
  this->scale_factor = scale_factor;
  this->inputs = new tensor<double> *[1];
  this->inputs[0] = inputs[0];
}

void Opsscale::initilizeoutput(tensor<double> *outputs) {
  this->output = outputs;
  *(this->output) = *(inputs[0]);
}

void Opsscale::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsscale::printoutput() {
  // std::cout << output->getData() << "\n";
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}
