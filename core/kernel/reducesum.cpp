#include <LAS/CPULibrary.h>
#include <LAS/avx2_micro_kernels.h>
#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>


void Opsreducesum::recursive_sum(unsigned index, unsigned *dimension_arr,
                                 unsigned reduction_dim,
                                 std::float64_t *temp_arr) {

  if (index < 3) {
    unsigned i, j, k;
    unsigned x_axis, y_axis, z_axis, stride, n_dim_size;
    unsigned input_index, output_index;
    std::float64_t *input_ptr, *output_ptr, *temp_inp;
    std::float64_t *ptr[3];
    unsigned a[2];

    x_axis = temp_input->getDimensions()[0];
    y_axis = (temp_input->getNoOfDimensions() > 1) ? temp_input->getDimensions()[1] : 1;
    z_axis = (temp_input->getNoOfDimensions() > 2) ? temp_input->getDimensions()[2] : 1;

    input_ptr = temp_input->getData();
    output_ptr = temp_output->getData();

    input_index = output_index = 0;

    if (temp_input->getNoOfDimensions() > 3) {
      n_dim_size = x_axis * y_axis * z_axis;
      for (i = 3; i < temp_input->getNoOfDimensions(); i++) {
        input_index += n_dim_size * dimension_arr[i];
        n_dim_size *= temp_input->getDimensions()[i];
      }

      n_dim_size = 1;
      for (i = 0; i < temp_input->getNoOfDimensions(); i++) {
        if (i != reduction_dim) {
          if (i < 3)
            output_index *= n_dim_size;
          else
            output_index += n_dim_size * dimension_arr[i];

          n_dim_size *= temp_input->getDimensions()[i];
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
          temp_arr[i + j * y_axis] =
                input_ptr[i * x_axis + j * x_axis * y_axis + stride * k + input_index];

        ptr[1] = temp_arr;
        cpu::__madd(ptr, a);
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
          temp_arr[i + j * x_axis] =
                input_ptr[i + j * x_axis * y_axis + stride * k + input_index];

        ptr[1] = temp_arr;
        cpu::__madd(ptr, a);
      }

      break;
    }
    case 2: {

      ptr[0] = ptr[2] = output_ptr + output_index;
      a[0] = x_axis;
      a[1] = y_axis;

      for (k = 0; k < z_axis; k++) {
        stride = x_axis * y_axis;
        temp_arr = input_ptr + (stride * k + input_index);
        ptr[1] = temp_arr;

        cpu::__madd(ptr, a);
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
      }
      break;
    }
    }
  } else {
    for (unsigned i = 0; i < temp_input->getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_sum(index - 1, dimension_arr, reduction_dim, temp_arr);
    }
  }
}

void Opsreducesum::compute() {
  unsigned i, k, resulting_no_of_dims;

  unsigned *resulting_dims, *arr_dims;
  std::float64_t *intermediate_input;
  Tensor<std::float64_t> temp_inp = Tensor<std::float64_t>(*inputs[0]);
  temp_inp.initData(inputs[0]->getData());
  temp_inp.printData();

  intermediate_input = new std::float64_t[inputs[0]->getDimensions()[0] *
                                          inputs[0]->getDimensions()[1] *
                                          inputs[0]->getDimensions()[2]];

  resulting_dims = new unsigned(inputs[0]->getNoOfDimensions());
  arr_dims = new unsigned(inputs[0]->getNoOfDimensions());
  temp_output = inputs[0];
  temp_input = &temp_inp;

  for (i = 0; i < no_of_reduction_dim; i++) {
    resulting_no_of_dims = temp_output->getNoOfDimensions()
                               ? temp_output->getNoOfDimensions() - 1
                               : 1;
    if (temp_output->getNoOfDimensions() > 1) {
      k = 0;
      for (unsigned j = 0; j < temp_output->getNoOfDimensions(); j++)
        if (j != reduction_dims[i])
          resulting_dims[k++] = temp_output->getDimensions()[j];
    } else {
      resulting_no_of_dims = 1;
      resulting_dims[0] = 1;
    }

    (*temp_output) = Tensor<std::float64_t>(resulting_no_of_dims, resulting_dims,
                                         inputs[0]->getType());
    temp_output->initData(0.0);

    recursive_sum(temp_input->getNoOfDimensions() - 1, arr_dims,
                  reduction_dims[i], intermediate_input);

    temp_input = temp_output;
  }
  output->initData(temp_output->getData());
}

void Opsreducesum::initilizeinputs(Tensor<std::float64_t> **inputs, unsigned n,
                                   unsigned *arr) {
  unsigned i;
  this->no_of_reduction_dim = n;

  reduction_dims = new unsigned[n];
  for (i = 0; i < n; i++)
    reduction_dims[i] = arr[i];

  this->inputs = new Tensor<std::float64_t> *[1];
  this->inputs[0] = inputs[0];
}

void Opsreducesum::initilizeoutput(Tensor<std::float64_t> *output) {
  unsigned no_of_resultent_dims, *resultent_dims;
  unsigned i, j;
  this->output = output;

  no_of_resultent_dims = inputs[0]->getNoOfDimensions() - no_of_reduction_dim;
  resultent_dims = new unsigned[no_of_reduction_dim];

  j = 0;
  for (unsigned i = 0; i < inputs[0]->getNoOfDimensions(); i++)
    if (i != reduction_dims[j])
      resultent_dims[j++] = inputs[0]->getDimensions()[i];

  *(this->output) = Tensor<std::float64_t>(no_of_resultent_dims, resultent_dims,
                                           inputs[0]->getType());
}

void Opsreducesum::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsreducesum::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}
