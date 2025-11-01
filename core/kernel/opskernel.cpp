#include "opskernel.h"
#include "../framework/MathLibrary.h"


Ops::~Ops() {}

void Ops::recursive_iterator(unsigned index, unsigned *dimension_arr,
                             // Tensor<std::float64_t> input_a,
                             Tensor<std::float64_t> input_b, Tensor<std::float64_t> &output,
                             std::string function_name, unsigned *ui_arr,
                             std::float64_t *dl_arr, Tensor<std::float64_t> *misc_arr) {}

void Ops::recursive_sum(unsigned index, unsigned *dimension_arr,
                        Tensor<std::float64_t> input_b, Tensor<std::float64_t> &output,
                        unsigned reduction_dim, std::float64_t *temp_input) {

  if (index < 3) {
    unsigned i, j, k;
    unsigned x_axis, y_axis, z_axis, stride, n_dim_size;
    unsigned input_index, output_index;
    std::float64_t *input_ptr, *output_ptr, *temp_inp;
    std::float64_t *ptr[3];
    unsigned a[2];

    // std::float64_t *input_b = input_a.getData();

    x_axis = input_b.getDimensions()[0];
    y_axis = (input_b.getNoOfDimensions() > 1) ? input_b.getDimensions()[1] : 1;
    z_axis = (input_b.getNoOfDimensions() > 2) ? input_b.getDimensions()[2] : 1;

    input_ptr = input_b.getData();
    output_ptr = output.getData();

    input_index = output_index = 0;

    if (input_b.getNoOfDimensions() > 3) {
      n_dim_size = x_axis * y_axis * z_axis;
      for (i = 3; i < input_b.getNoOfDimensions(); i++) {
        input_index += n_dim_size * dimension_arr[i];
        n_dim_size *= input_b.getDimensions()[i];
      }

      n_dim_size = 1;
      for (i = 0; i < input_b.getNoOfDimensions(); i++) {
        if (i != reduction_dim) {
          if (i < 3)
            output_index *= n_dim_size;
          else
            output_index += n_dim_size * dimension_arr[i];

          n_dim_size *= input_b.getDimensions()[i];
        }
      }
    }

    switch (reduction_dim) {
    case 0: {

      ptr[0] = ptr[2] = output_ptr + output_index;
      a[0] = x_axis;
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
    for (unsigned i = 0; i < input_b.getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_sum(index - 1, dimension_arr, input_b, output, reduction_dim,
                    temp_input);
    }
  }
}
