#include "opskernel.h"
#include "../framework/MathLibrary.h"

void Opsreducesum::recursive_sum(
    unsigned index, unsigned *dimension_arr, tensor<double> input,
    tensor<double> &output, unsigned reduction_dim, double *temp_input) {

  if (index < 3) {
    unsigned i, j, k;
    unsigned x_axis, y_axis, z_axis, stride, n_dim_size;
    unsigned input_index, output_index;
    double *input_ptr, *output_ptr, *temp_inp;
    double *ptr[3];
    unsigned a[2];

    // double *input = this->getData();

    x_axis = input.getDimensions()[0];
    y_axis = (input.getNoOfDimensions() > 1) ? input.getDimensions()[1] : 1;
    z_axis = (input.getNoOfDimensions() > 2) ? input.getDimensions()[2] : 1;

    input_ptr = input.getData();
    output_ptr = output.getData();

    input_index = output_index = 0;

    if (input.getNoOfDimensions() > 3) {
      n_dim_size = x_axis * y_axis * z_axis;
      for (i = 3; i < input.getNoOfDimensions(); i++) {
        input_index += n_dim_size * dimension_arr[i];
        n_dim_size *= input.getDimensions()[i];
      }

      n_dim_size = 1;
      for (i = 0; i < input.getNoOfDimensions(); i++) {
        if (i != reduction_dim) {
          if (i < 3)
            output_index *= n_dim_size;
          else
            output_index += n_dim_size * dimension_arr[i];

          n_dim_size *= input.getDimensions()[i];
        }
      }
    }

    switch (reduction_dim) {
    case 0: {

      ptr[0] = ptr[2] = output_ptr + output_index;
      a[0] = y_axis;
      a[1] = z_axis;

      for (k = 0; k < x_axis; k++) {
        stride = 1;
        for (j = 0; j < z_axis; j++)
          for (i = 0; i < y_axis; i++)
            temp_input[i + j * y_axis] =
                input_ptr[i * x_axis + j * x_axis * y_axis + stride * k +
                          input_index];

        ptr[1] = temp_input;
        cpu::__madd(ptr, a);
        // cpu::__madd(output_ptr + output_index, temp_input, output_ptr +
        // output_index, y_axis, z_axis);
      }
      break;
    }

    case 1: {

      ptr[0] = ptr[2] = output_ptr + output_index;
      a[0] = x_axis;
      a[1] = z_axis;
      for (k = 0; k < y_axis; k++) {
        stride = x_axis;
        for (j = 0; j < z_axis; j++)
          for (i = 0; i < x_axis; i++)
            temp_input[i + j * x_axis] =
                input_ptr[i + j * x_axis * y_axis + stride * k + input_index];

        ptr[1] = temp_input;
        cpu::__madd(ptr, a);
        // cpu::__madd(output_ptr + output_index, temp_input, output_ptr +
        // output_index, x_axis, z_axis);
      }

      break;
    }
    case 2: {

      ptr[0] = ptr[2] = output_ptr + output_index;
      a[0] = x_axis;
      a[1] = y_axis;

      for (k = 0; k < z_axis; k++) {
        stride = x_axis * y_axis;
        temp_input = input_ptr + (stride * k + input_index);
        ptr[1] = temp_input;

        cpu::__madd(ptr, a);

        // for (int j = 0; j < y_axis; j++)
        //     for (int i = 0; i < x_axis; i++)
        //         std::cout << output_ptr[i + j * x_axis] << " ";

        // cpu::__madd(output_ptr + output_index, temp_inp, output_ptr +
        // output_index, x_axis, y_axis);
      }
      break;
    }

    default: {
      a[0] = x_axis;
      a[1] = y_axis;
      for (k = 0; k < z_axis; k++) {
        stride = x_axis * y_axis;

        ptr[0] = ptr[2] = output_ptr + (output_index + stride * k);

        temp_inp = input_ptr + (stride * k + input_index);
        ptr[1] = temp_inp;

        cpu::__madd(ptr, a);
        // cpu::__madd(output_ptr + (output_index + stride * k), temp_inp,
        // output_ptr + (output_index + stride * k), x_axis, y_axis);
      }
      break;
    }
    }
  } else {
    for (unsigned i = 0; i < input.getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_sum(index - 1, dimension_arr, input, output, reduction_dim,
                    temp_input);
    }
  }
}


void Opsreducesum::compute()  {
  unsigned i, k, resulting_no_of_dims;

  unsigned *resulting_dims, *arr_dims;
  double *intermediate_input;
  tensor<double> temp_output, temp_input;

  intermediate_input =
      new double[inputs[0]->getDimensions()[0] * inputs[0]->getDimensions()[1] *
            inputs[0]->getDimensions()[2]];

  resulting_dims = new unsigned(inputs[0]->getNoOfDimensions());
  arr_dims = new unsigned(inputs[0]->getNoOfDimensions());
  temp_output = *(inputs[0]);
  temp_input = *(inputs[0]);

  for (i = 0; i < no_of_reduction_dim; i++) {
    resulting_no_of_dims = temp_output.getNoOfDimensions()
                               ? temp_output.getNoOfDimensions() - 1
                               : 1;
    if (temp_output.getNoOfDimensions() > 1) {
      k = 0;
      for (unsigned j = 0; j < temp_output.getNoOfDimensions(); j++)
        if (j != reduction_dims[i])
          resulting_dims[k++] = temp_output.getDimensions()[j];
    } else {
      resulting_no_of_dims = 1;
      resulting_dims[0] = 1;
    }

    temp_output = tensor<double>(resulting_no_of_dims, resulting_dims);
    temp_output.initData(0.0);

    recursive_sum(temp_input.getNoOfDimensions() - 1, arr_dims, temp_input,
                  temp_output, reduction_dims[i], intermediate_input);

    temp_input = temp_output;
  }
  output->initData(temp_output);
}


void Opsreducesum::initilizeinputs(tensor<double> **inputs,
                                                unsigned n,
                                                unsigned *arr)  {
  unsigned i;
  this->no_of_reduction_dim = n;

  reduction_dims = new unsigned[n];
  for (i = 0; i < n; i++)
    reduction_dims[i] = arr[i];

  for (i = 0; i < n; i++)
    std::cout << reduction_dims[i] << ", ";
  std::cout << "\n";

  this->inputs = new tensor<double> *[1];

  this->inputs[0] = inputs[0];
}


void Opsreducesum::initilizeoutput(
    tensor<double> *output)  {
  unsigned no_of_resultent_dims, *resultent_dims;
  unsigned i, j;
  this->output = output;

  no_of_resultent_dims = inputs[0]->getNoOfDimensions() - no_of_reduction_dim;
  resultent_dims = new unsigned[no_of_reduction_dim];

  j = 0;

  for (unsigned i = 0; i < inputs[0]->getNoOfDimensions(); i++)
    if (i != reduction_dims[j])
      resultent_dims[j++] = inputs[0]->getDimensions()[i];

  *(this->output) = tensor<double>(no_of_resultent_dims, resultent_dims);
}


void Opsreducesum::printinputs()  {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}


void Opsreducesum::printoutput()  {
  // std::cout << output->getData() << "\n";
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}
