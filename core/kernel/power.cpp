#include "opskernel.h"
#include "../framework/MathLibrary.h"

void Opspower::recursive_iterator(
    unsigned index, unsigned *dimension_arr, tensor<double> input_a,
    tensor<double> input_b, tensor<double> &output,
    std::string function_name, unsigned *ui_arr, double *dl_arr,
    tensor<double> *misc_arr) {
  if (index < 2) {
    unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
    unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
        c_index;

    inpA_x = (input_a.getNoOfDimensions() > 0) ? input_a.getDimensions()[0] : 1;
    inpA_y = (input_a.getNoOfDimensions() > 1) ? input_a.getDimensions()[1] : 1;

    inpB_x = (input_b.getNoOfDimensions() > 0) ? input_b.getDimensions()[0] : 1;
    inpB_y = (input_b.getNoOfDimensions() > 1) ? input_b.getDimensions()[1] : 1;

    out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
    out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

    a_plane_size = inpA_x * inpA_y;
    b_plane_size = inpB_x * inpB_y;
    c_plane_size = out_x * out_y;

    a_index = b_index = c_index = 0;
    if (input_b.getNoOfDimensions() > 2)
      for (i = 2; i < input_b.getNoOfDimensions(); i++) {
        a_index += a_plane_size * dimension_arr[i];
        b_index += b_plane_size * dimension_arr[i];
        c_index += c_plane_size * dimension_arr[i];

        a_plane_size *= input_a.getDimensions()[i];
        b_plane_size *= input_b.getDimensions()[i];
        c_plane_size *= output.getDimensions()[i];
      }

      int j;
      // int exponent = ui_arr[0];
      unsigned a[2];
      double *ptr[3];

      a[0] = inpA_x;
      a[1] = inpA_y;

      ptr[0] = input_a.getData() + a_index;
      ptr[1] = input_b.getData() + b_index;
      ptr[2] = output.getData() + c_index;

      cpu::__melementwisemul(ptr, a);

  } else {
    // std::cout << "inside else\n";
    for (unsigned i = 0; i < input_a.getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, input_a, input_b, output,
                         function_name, NULL, NULL, NULL);
    }
  }
};


void Opspower::compute()  {
  unsigned i, *arr;

  if (exponent == 0)
    output->initData(1);
  else if (exponent > 0) {
    output->initData(inputs[0]->getData());
    arr = new unsigned[inputs[0]->getNoOfDimensions()];

    // std::cout << output.getData() << "\n";
    for (i = 1; i < exponent; i++)
      recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, *(output),
                         *(inputs[0]), *(output), "matrix_power", NULL, NULL,
                         NULL);
    delete[] arr;
  }
}


void Opspower::initilizeinputs(tensor<double> **inputs, unsigned exponent)  {
  unsigned i;
  this->exponent = exponent;

  this->inputs = new tensor<double> *[1];

  for (i = 0; i < 1; i++) {
    this->inputs[i] = inputs[i];
  }
}


void Opspower::initilizeoutput(tensor<double> *output)  {
  this->output = output;

  *(this->output) = *(inputs[0]);
}


void Opspower::printinputs()  {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}


void Opspower::printoutput()  {
  // std::cout << output->getData() << "\n";
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

