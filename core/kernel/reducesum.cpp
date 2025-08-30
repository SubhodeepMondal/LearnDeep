#ifdef CUDA_ENABLED
#include <LAS/gpu_interface.cuh>
#endif

#include <LAS/CPULibrary.h>
#include <LAS/avx2_micro_kernels.h>
#include <absl/log/log.h>
#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>

Opsreducesum::~Opsreducesum() {
  delete temp_input;
  delete temp_output;
}
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
    y_axis = (temp_input->getNoOfDimensions() > 1)
                 ? temp_input->getDimensions()[1]
                 : 1;
    z_axis = (temp_input->getNoOfDimensions() > 2)
                 ? temp_input->getDimensions()[2]
                 : 1;

    input_ptr = temp_input->getData();
    output_ptr = temp_output->getData();

    input_index = output_index = 0;

    if (temp_input->getNoOfDimensions() > 3) {
      n_dim_size = x_axis * y_axis * z_axis;
      // Calculate the input index based on the dimensions
      for (i = 3; i < temp_input->getNoOfDimensions(); i++) {
        input_index += n_dim_size * dimension_arr[i];
        n_dim_size *= temp_input->getDimensions()[i];
      }

      n_dim_size = 1;
      // Calculate the output index based on the dimensions
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
                input_ptr[i * x_axis + j * x_axis * y_axis + stride * k +
                          input_index];

        ptr[1] = temp_arr;

        kernel_dispatch(ptr, a);
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

        kernel_dispatch(ptr, a);
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

        kernel_dispatch(ptr, a);
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
        kernel_dispatch(ptr, a);
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
  unsigned nElements = 1;
  for (i = 0; i < this->inputs[0]->getNoOfDimensions() && i < 3; i++)
    nElements *= this->inputs[0]->getDimensions()[i];
  std::float64_t *intermediate_input = new std::float64_t[nElements];

  resulting_dims = new unsigned[inputs[0]->getNoOfDimensions()];
  arr_dims = new unsigned[inputs[0]->getNoOfDimensions()];
  temp_input = new Tensor<std::float64_t>(*this->inputs[0]);
  temp_output = new Tensor<std::float64_t>(*this->output);
  temp_input->initData(this->inputs[0]->getData());

  for (i = 0; i < no_of_reduction_dim; i++) {
    LOG(INFO) << "Reducing on dimension: " << reduction_dims[i] - i << "\n";
    resulting_no_of_dims = temp_input->getNoOfDimensions() - 1;

    if (temp_input->getNoOfDimensions() > 1) {
      k = 0;
      for (unsigned j = 0; j < temp_input->getNoOfDimensions(); j++)
        if (j != reduction_dims[i] - i)
          resulting_dims[k++] = temp_input->getDimensions()[j];
    } else {
      resulting_no_of_dims = 1;
      resulting_dims[0] = 1;
    }

    temp_output->reshape(resulting_no_of_dims, resulting_dims);
    temp_output->initData(0.0);

    recursive_sum(temp_input->getNoOfDimensions() - 1, arr_dims,
                  reduction_dims[i] - i, intermediate_input);

    temp_input->reshape(resulting_no_of_dims, resulting_dims);
    temp_input->initData(temp_output->getData());
  }
  this->output->initData(temp_output->getData());
  delete[] intermediate_input;
  delete[] resulting_dims;
  delete[] arr_dims;
}

void Opsreducesum::initilizeinputs(Tensor<std::float64_t> **inputs, unsigned n,
                                   unsigned *arr) {
  unsigned i;
  this->no_of_reduction_dim = n;

  // reduction_dims = new unsigned[n];
  for (i = 0; i < n; i++)
    reduction_dims.push_back(arr[i]);

  this->inputs[0] = inputs[0];
}

void Opsreducesum::initilizeoutput(Tensor<std::float64_t> *output) {
  unsigned no_of_resultent_dims;
  std::vector<unsigned> resultent_dims;
  unsigned i, j;
  this->output = output;

  no_of_resultent_dims = inputs[0]->getNoOfDimensions() - no_of_reduction_dim;
  // resultent_dims = new unsigned[no_of_reduction_dim];

  j = 0;
  for (unsigned i = 0; i < inputs[0]->getNoOfDimensions(); i++)
    if (i != reduction_dims[j])
      resultent_dims.push_back(inputs[0]->getDimensions()[i]);

  this->output->reshape(no_of_resultent_dims, resultent_dims.data());
}

void Opsreducesum::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    LOG(INFO) << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsreducesum::printoutput() {
  LOG(INFO) << "output:\n";
  output->printData();
  LOG(INFO) << "\n";
}

void Opsreducesum::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {

#ifdef CUDA_ENABLED
  double *d_arr[3];
  d_arr[0] = reinterpret_cast<double *>(ptr[0]);
  d_arr[1] = reinterpret_cast<double *>(ptr[1]);
  d_arr[2] = reinterpret_cast<double *>(ptr[2]);

  gpu::gpu_mat_add_f64(d_arr, arr);
#else
  if (__builtin_cpu_supports("avx2")) {
    avx2::avx2_add_f64(ptr, arr);
  } else {
    cpu::__madd(ptr, arr);
  }
#endif
}
