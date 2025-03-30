#include "../framework/MathLibrary.h"
#include "opskernel.h"

void Opsmatmul::recursive_iterator(unsigned index, unsigned *dimension_arr,
                                   tensor<double> input_a,
                                   tensor<double> input_b,
                                   tensor<double> &output,
                                   std::string function_name, unsigned *ui_arr,
                                   double *dl_arr, tensor<double> *misc_arr) {
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

    /* code */
    unsigned a[3];
    double *ptr[3];

    a[0] = inpB_x;
    a[1] = inpB_y;
    a[2] = inpA_y;

    ptr[0] = input_a.getData() + a_index;
    ptr[1] = input_b.getData() + b_index;
    ptr[2] = output.getData() + c_index;
    cpu::__mmul(ptr, a);

  } else {
    // std::cout << "inside else\n";
    for (unsigned i = 0; i < input_a.getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, input_a, input_b, output,
                         function_name, NULL, NULL, NULL);
    }
  }
};

void Opsmatmul::compute() {
  // std::cout << "inside compute\n";
  unsigned dim_x, dim_y;
  dim_x = inputs[0]->getDimensions()[0];
  dim_y = inputs[1]->getDimensions()[1];

  unsigned *arr = new unsigned[inputs[0]->getNoOfDimensions()];

  recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, *inputs[0],
                     *inputs[1], *output, "matrix_multiplication", NULL, NULL,
                     NULL);

  delete[] arr;
}

void Opsmatmul::initilizeinputs(tensor<double> **inputs,
                                unsigned no_of_inputs) {
  unsigned i;
  this->no_of_inputs = no_of_inputs;

  this->inputs = new tensor<double> *[this->no_of_inputs];

  for (i = 0; i < this->no_of_inputs; i++) {
    this->inputs[i] = inputs[i];
  }
}

void Opsmatmul::initilizeoutput(tensor<double> *output) {
  this->output = output;

  unsigned i, dim_x, dim_y, no_of_dimensions;
  unsigned *output_dim;

  no_of_dimensions = inputs[0]->getNoOfDimensions();

  dim_x = inputs[1]->getDimensions()[0];
  dim_y = inputs[0]->getDimensions()[1];

  output_dim = new unsigned[no_of_dimensions];

  output_dim[0] = dim_x;
  output_dim[1] = dim_y;

  for (i = 2; i < no_of_dimensions; i++)
    output_dim[i] = inputs[0]->getDimensions()[i];

  *(this->output) = tensor<double>(no_of_dimensions, output_dim);

  delete[] output_dim;
}

void Opsmatmul::printinputs() {
  unsigned i;
  for (i = 0; i < this->no_of_inputs; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsmatmul::printoutput() {
  // std::cout << output->getData() << "\n";
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}
