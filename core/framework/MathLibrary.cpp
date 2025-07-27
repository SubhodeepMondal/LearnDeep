#include "MathLibrary.h"
#include "../kernel/opskernel.h"
template <typename T>
void Tensor<T>::recursive_iterator(unsigned index, unsigned *dimension_arr,
                                   Tensor<T> input, Tensor<T> &output,
                                   void (*__kernel)(T **, unsigned *),
                                   std::string function_name, unsigned *ui_arr,
                                   T *dl_arr, Tensor<T> misc_arr) {
  if (index < 2) {
    unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
    unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
        c_index;

    inpA_x =
        (this->getNoOfDimensions() > 0) ? Tensor<T>::getDimensions()[0] : 1;
    inpA_y =
        (this->getNoOfDimensions() > 1) ? Tensor<T>::getDimensions()[1] : 1;

    out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
    out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

    a_plane_size = inpA_x * inpA_y;
    c_plane_size = out_x * out_y;

    a_index = b_index = c_index = 0;

    if (this->getNoOfDimensions() > 2)
      for (i = 2; i < this->getNoOfDimensions(); i++) {
        a_index += a_plane_size * dimension_arr[i];
        c_index += c_plane_size * dimension_arr[i];

        a_plane_size *= this->getDimensions()[i];
        c_plane_size *= output.getDimensions()[i];
      }

    switch (fx_name) {
    case matrix_multiplication: {
      /* code */
      unsigned a[3];
      T *ptr[3];

      inpB_x = (input.getNoOfDimensions() > 0) ? input.getDimensions()[0] : 1;
      inpB_y = (input.getNoOfDimensions() > 1) ? input.getDimensions()[1] : 1;
      b_plane_size = inpB_x * inpB_y;

      if (input.getNoOfDimensions() > 2)
        for (i = 2; i < input.getNoOfDimensions(); i++) {
          b_index += b_plane_size * dimension_arr[i];
          b_plane_size *= input.getDimensions()[i];
        }

      a[0] = inpB_x;
      a[1] = inpB_y;
      a[2] = inpA_y;

      ptr[0] = Tensor<T>::getData() + a_index;
      ptr[1] = input.getData() + b_index;
      ptr[2] = output.getData() + c_index;
      __kernel(ptr, a);

      break;
    }
    case matrix_addition: {
      /* code */
      unsigned a[2];
      T *ptr[3];

      inpB_x = (input.getNoOfDimensions() > 0) ? input.getDimensions()[0] : 1;
      inpB_y = (input.getNoOfDimensions() > 1) ? input.getDimensions()[1] : 1;
      b_plane_size = inpB_x * inpB_y;

      if (input.getNoOfDimensions() > 2)
        for (i = 2; i < input.getNoOfDimensions(); i++) {
          b_index += b_plane_size * dimension_arr[i];
          b_plane_size *= input.getDimensions()[i];
        }

      a[0] = inpA_x;
      a[1] = inpA_y;

      ptr[0] = Tensor<T>::getData() + a_index;
      ptr[1] = input.getData() + b_index;
      ptr[2] = output.getData() + c_index;
      __kernel(ptr, a);

      break;
    }
    case matrix_power: {
      unsigned a[2];
      T *ptr[3];

      inpB_x = (input.getNoOfDimensions() > 0) ? input.getDimensions()[0] : 1;
      inpB_y = (input.getNoOfDimensions() > 1) ? input.getDimensions()[1] : 1;
      b_plane_size = inpB_x * inpB_y;

      if (input.getNoOfDimensions() > 2)
        for (i = 2; i < input.getNoOfDimensions(); i++) {
          b_index += b_plane_size * dimension_arr[i];
          b_plane_size *= input.getDimensions()[i];
        }

      a[0] = inpA_x;
      a[1] = inpA_y;

      ptr[0] = this->getData() + a_index;
      ptr[1] = input.getData() + b_index;
      ptr[2] = output.getData() + c_index;

      // cpu::__melementwisemul(ptr, a);

      break;
    }
    case matrix_scaler_multiplication: {
      unsigned a[2];
      T *ptr[3];

      a[0] = inpA_x;
      a[1] = inpA_y;

      ptr[0] = this->getData() + a_index;
      ptr[1] = dl_arr;
      ptr[2] = output.getData() + c_index;

      // cpu::__mscalermul(ptr, a);

      break;
    }

    default:
      break;
    }
  } else {
    // std::cout << "inside else\n";
    for (unsigned i = 0; i < Tensor<T>::getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, input, output, __kernel,
                         function_name, ui_arr, dl_arr, misc_arr);
    }
  }
}

template <typename T>
void Tensor<T>::recursive_sum(unsigned index, unsigned *dimension_arr,
                              Tensor<T> input, Tensor<T> &output,
                              unsigned reduction_dim, T *temp_input) {

  if (index < 3) {
    unsigned i, j, k;
    unsigned x_axis, y_axis, z_axis, stride, n_dim_size;
    unsigned input_index, output_index;
    T *input_ptr, *output_ptr, *temp_inp;
    T *ptr[3];
    unsigned a[2];

    // T *input = this->getData();

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
        // cpu::__madd(ptr, a);
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
        // cpu::__madd(ptr, a);
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

        // cpu::__madd(ptr, a);
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

template <typename T> Tensor<T> Tensor<T>::matmul(const Tensor<T> input) {
  Tensor<T> output;
  unsigned i, j, no_of_dimensions, flag = 1;
  unsigned a_plan_dim, b_plan_dim, c_plan_dim;
  unsigned a_actual_index, b_actual_index, c_actual_index;
  unsigned dim_x, dim_y, dim_z;
  unsigned *output_dim;
  DataType d_type = tf_float64;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  dim_x = input.getDimensions()[0];
  dim_y = Tensor<T>::getDimensions()[1];
  dim_z = Tensor<T>::getDimensions()[0];

  if (no_of_dimensions == input.getNoOfDimensions()) {
    output_dim = new unsigned[no_of_dimensions];

    output_dim[0] = dim_x;
    output_dim[1] = dim_y;

    if (this->getDimensions()[0] == input.getDimensions()[1]) {

      for (i = 2; i < no_of_dimensions; i++) {
        output_dim[i] = Tensor<T>::getDimensions()[i];
        if (Tensor<T>::getDimensions()[i] != input.getDimensions()[i]) {
          flag = 0;
          break;
        }
      }
      if (flag) {

        output = Tensor<T>(no_of_dimensions, output_dim, d_type);
        unsigned *dimension_arr = new unsigned[this->getNoOfDimensions()];

        // recursive_iterator(this->getNoOfDimensions() - 1, dimension_arr,
        // input,
        //                    output, cpu::__mmul, "matrix_multiplication",
        //                    NULL, NULL, NULL);

        delete[] dimension_arr;
        // output.printData();
      } else {
        std::cout << "Error!" << i
                  << "th Dimension does not match with second matrix.\n";
        return output;
      }
    } else {
      std::cout << "Error! First matrix's row length does not match with "
                   "second matrix column length.\n";
      return output;
    }
  } else {
    std::cout << "Dimension mismatch, First matrix doesn't have same no of "
                 "dimension of second matrix.\n";
    return output;
  }

  return output;
}

template <typename T>
Tensor<T> Tensor<T>::scale(const std::float64_t scaleFactor) {
  Tensor<T> output;

  unsigned no_of_dims;
  unsigned *arr;
  T scale_factor = scaleFactor;
  DataType d_type = tf_float64;

  no_of_dims = Tensor<T>::getNoOfDimensions();

  arr = new unsigned[no_of_dims];

  output = Tensor<T>(no_of_dims, this->getDimensions(), d_type);

  // recursive_iterator(this->getNoOfDimensions() - 1, arr, NULL, output,
  //                    cpu::__madd, "matrix_scaler_multiplication", NULL,
  //                    &scale_factor, NULL);

  delete[] arr;

  return output;
}

template <typename T> Tensor<T> Tensor<T>::operator*(const Tensor<T> input) {
  Tensor<T> output;
  DataType d_type;
  unsigned i, j, no_of_dimensions, flag = 1;
  unsigned a_plan_dim, b_plan_dim, c_plan_dim;
  unsigned a_actual_index, b_actual_index, c_actual_index;
  unsigned dim_x, dim_y, dim_z;
  unsigned *output_dim;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  dim_x = input.getDimensions()[0];
  dim_y = Tensor<T>::getDimensions()[1];
  dim_z = Tensor<T>::getDimensions()[0];

  if (no_of_dimensions == input.getNoOfDimensions()) {
    output_dim = new unsigned[no_of_dimensions];

    output_dim[0] = dim_x;
    output_dim[1] = dim_y;

    std::cout << dim_x << " " << dim_y << "\n";

    for (i = 2; i < no_of_dimensions; i++) {
      output_dim[i] = Tensor<T>::getDimensions()[i];
      if (Tensor<T>::getDimensions()[i] != input.getDimensions()[i]) {
        flag = 0;
        break;
      }
    }
    if (flag && this->getDimensions()[0] == input.getDimensions()[1]) {

      output = Tensor<T>(no_of_dimensions, output_dim, d_type);

      if (no_of_dimensions < 3) {

        std::cout << "inside no_of_dim < 3\n";
        output.printDimensions();
        std::cout << "\n";
        std::cout << output.getDimensions()[0] << "\n";
        std::cout << output.getDimensions()[1] << "\n";
        // cpu::__mmul(Tensor<T>::getData(), input.getData(),
        // output.getData(), dim_x, dim_y, dim_z);
        // output.printData();
      } else {
        a_plan_dim = dim_y * dim_z;
        b_plan_dim = dim_x * dim_z;
        c_plan_dim = dim_x * dim_y;
        a_actual_index = b_actual_index = c_actual_index = 0;
        for (i = 2; i < no_of_dimensions; i++) {
          for (j = 0; j < Tensor<T>::getDimensions()[i]; j++) {
            // cpu::__mmul(Tensor<T>::getData() + a_actual_index,
            //             input.getData() + b_actual_index,
            //             output.getData() + c_actual_index, dim_x, dim_y,
            //             dim_z);
            a_actual_index += a_plan_dim;
            b_actual_index += b_plan_dim;
            c_actual_index += c_plan_dim;
          }
        }

        std::cout << "inside no_of_dim >= 3\n";
      }
    }
    output.printData();
  }

  output.printData();
  return output;
}

template <typename T> Tensor<T> Tensor<T>::add(const Tensor<T> input) {
  Tensor<T> output;
  DataType d_type = tf_float64;

  unsigned dim_x, dim_y, plane_offset, no_of_dimensions, flag;

  flag = 1;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  for (int i = 0; i < no_of_dimensions; i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    dim_x = this->getDimensions()[0];
    dim_y = this->getDimensions()[1];
    plane_offset = 0;

    output =
        Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    unsigned *arr = new unsigned[this->getNoOfDimensions()];

    // recursive_iterator(this->getNoOfDimensions() - 1, arr, input, output,
    //                    cpu::__madd, "matrix_addition", NULL, NULL, NULL);
    return output;
  } else {
    std::cout << "Two metrix requires same shape to perform matrix addition, "
                 "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
    return output;
  }
}

template <typename T> Tensor<T> Tensor<T>::vectoradd(const Tensor<T> input) {
  Tensor<T> output, temp_input;

  unsigned dim_x, dim_y, plane_offset, no_of_dimensions, flag;
  DataType d_type = tf_float64;

  flag = 1;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  if (this->getDimensions()[0] != input.getDimensions()[0]) {
    std::cout << "Two metrix requires same shape for x-axis to perform matrix "
                 "addition, here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape on x-axis.\n";

    return output;
  } else {
    dim_x = this->getDimensions()[0];
    dim_y = this->getDimensions()[1];
    plane_offset = 0;

    output =
        Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    temp_input = Tensor<T>(dim_x, &dim_y, d_type);

    for (unsigned i = 0; i < dim_y; i++)
      temp_input.initPartialData(i * dim_x, dim_x, input.getData());

    if (no_of_dimensions < 3) {
      // cpu::__madd(this->getData(), temp_input.getData(), output.getData(),
      //             dim_x, dim_y);
    } else {
      for (int i = 2; i < no_of_dimensions; i++)
        for (int j = 0; j < this->getDimensions()[i]; j++) {
          // cpu::__madd(this->getData() + plane_offset,
          //             temp_input.getData() + plane_offset,
          //             output.getData() + plane_offset, dim_x, dim_y);
          plane_offset += dim_x * dim_y;
        }
    }
    temp_input.destroy();
    return output;
  }
}

template <typename T> Tensor<T> Tensor<T>::operator+(const Tensor<T> input) {
  Tensor<T> output;

  unsigned dim_x, dim_y, plane_offset, no_of_dimensions, flag;
  DataType d_type = tf_float64;

  flag = 1;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  for (int i = 0; i < no_of_dimensions; i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    dim_x = this->getDimensions()[0];
    dim_y = this->getDimensions()[1];
    plane_offset = 0;

    output =
        Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);

    if (no_of_dimensions < 3) {
      // cpu::__madd(this->getData(), input.getData(), output.getData(), dim_x,
      // dim_y);
    } else {
      for (int i = 2; i < no_of_dimensions; i++)
        for (int j = 0; j < this->getDimensions()[i]; j++) {
          // cpu::__madd(this->getData() + plane_offset, input.getData() +
          // plane_offset, output.getData() + plane_offset, dim_x, dim_y);
          plane_offset += dim_x * dim_y;
        }
    }
    return output;
  } else {
    std::cout << "Two metrix requires same shape to perform matrix addition, "
                 "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
    return output;
  }
}

template <typename T> Tensor<T> Tensor<T>::operator-(const Tensor<T> input) {
  Tensor<T> output;
  DataType d_type = tf_float64;

  unsigned dim_x, dim_y, plane_offset, no_of_dimensions, flag;

  flag = 1;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  for (int i = 0; i < no_of_dimensions; i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    dim_x = this->getDimensions()[0];
    dim_y = this->getDimensions()[1];
    plane_offset = 0;

    output =
        Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);

    unsigned *dimension_arr = new unsigned[this->getNoOfDimensions()];

    // recursive_iterator(this->getNoOfDimensions() - 1, dimension_arr, input,
    //                    output, cpu::__msub, "matrix_multiplication", NULL,
    //                    NULL, NULL);

    // if (no_of_dimensions < 3)
    // {
    //     cpu::__madd(this->getData(), input.getData(), output.getData(),
    //     dim_x, dim_y);
    // }
    // else
    // {
    //     for (int i = 2; i < no_of_dimensions; i++)
    //         for (int j = 0; j < this->getDimensions()[i]; j++)
    //         {
    //             cpu::__msub(this->getData() + plane_offset, input.getData() +
    //             plane_offset, output.getData() + plane_offset, dim_x, dim_y);
    //             plane_offset += dim_x * dim_y;
    //         }
    // }

    return output;
  } else {
    std::cout << "Two metrix requires same shape to perform matrix addition, "
                 "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
    return output;
  }
}

template <typename T> void Tensor<T>::transpose() {
  unsigned x, y;

  x = Tensor<T>::getDimensions()[0];
  y = Tensor<T>::getDimensions()[1];

  // Tensor<T>::reshape(y, x);

  x = x + y;
  y = x - y;
  x = x - y;

  // cpu::__mtranspose(this->getData(), this->getData(), x, y);
}

template <typename T> void Tensor<T>::reducesum(Tensor<T> &output) {

  unsigned count = 0;
  unsigned *reduction_dims;
  unsigned *dims;
  Tensor<T> temp_output, temp_input;
  T *intermediate_input;

  arr_dims = new unsigned[this->getNoOfDimensions()];

  ptr = ptr_prev = head;

  while (ptr) {
    count++;
    ptr = ptr->next;
  }

  reduction_dims = new unsigned[count];
  ptr = head;

  for (unsigned i = 0; i < count; i++) {
    reduction_dims[i] = this->getNoOfDimensions() - ptr->value - 1;
    ptr = ptr->next;
    delete[] ptr_prev;
    ptr_prev = ptr;
  }

  // shorting dimensions using bubble short
  for (unsigned j = 0; j < count; j++)
    for (unsigned i = 0; i < count - j - 1; i++)
      if (reduction_dims[i] < reduction_dims[i + 1]) {
        unsigned temp = reduction_dims[i];
        reduction_dims[i] = reduction_dims[i + 1];
        reduction_dims[i + 1] = temp;
      }

  // unsigned resulting_dim = this->getNoOfDimensions() - count;

  unsigned resulting_no_of_dims, flag, k;
  unsigned *resulting_dims;
  DataType d_type = tf_float64;

  temp_output =
      Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  temp_input =
      Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);

  temp_output.setObjName("temp_output");
  temp_input.setObjName("temp_input");

  temp_input.initData(this->getData());

  intermediate_input =
      new T[this->getDimensions()[0] * this->getDimensions()[1] *
            this->getDimensions()[2]];

  resulting_dims = new unsigned[this->getNoOfDimensions()];

  for (unsigned i = 0; i < count; i++) {

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

    // temp_output.destroy();
    temp_output = Tensor<T>(resulting_no_of_dims, resulting_dims, d_type);
    temp_output.initData(0.0);

    std::cout << "Reducing dimension:" << reduction_dims[i] << "\n";

    recursive_sum(temp_input.getNoOfDimensions() - 1, arr_dims, temp_input,
                  temp_output, reduction_dims[i], intermediate_input);

    temp_input = temp_output;
  }

  output = temp_output;

  /*

  This destroy is not necessary, but it is here to fix a bug.
  when we're setting A Tensor of dim 3 and each dimension as (4, 3, 4) (last
  dimension >3) and doing a reduction sum on all dimensions its destroyer
  throwing an error! Error: free(): invalid next size (fast):
  */
  temp_output.destroy();

  delete[] resulting_dims;
  delete[] intermediate_input;
}

template <typename T>
template <typename first_dim, typename... Args>
void Tensor<T>::reducesum(Tensor<T> &output, first_dim n, Args... args) {
  if (n < this->getNoOfDimensions()) {

    if (!head) {
      ptr = new struct arg_list;

      head = ptr_prev = ptr;

      head->value = n;
      head->next = NULL;
    } else {
      ptr = new struct arg_list;

      ptr->value = n;
      ptr->next = NULL;
      ptr_prev->next = ptr;
      ptr_prev = ptr;
    }
    reducesum(output, args...);
  } else
    std::cout << "Fatal error! reduction axis does not belong for the Tensor\n";
}

template <typename T>
template <typename... Args>
Tensor<T> Tensor<T>::reducesum(Args... args) {
  head = NULL;
  Tensor<T> output;

  std::cout << "in normal reducesum!.";
  reducesum(output, args...);
  return output;
}

template <typename T> Tensor<T> Tensor<T>::pow(const unsigned exponent) {
  unsigned i, *arr;
  DataType d_type = tf_float64;

  Tensor<T> output(this->getNoOfDimensions(), this->getDimensions(), d_type);

  if (exponent == 0)
    output.initData(1);
  else if (exponent > 0) {
    output.initData(this->getData());
    arr = new unsigned[this->getNoOfDimensions()];

    // std::cout << output.getData() << "\n";
    for (i = 1; i < exponent; i++)
      // recursive_iterator(this->getNoOfDimensions() - 1, arr, output, output,
      //                    cpu::__melementwisemul, "matrix_power", NULL, NULL,
      //                    NULL);
      delete[] arr;
  }

  return output;
}

template <typename T> Ops *Tensor<T>::add(Tensor<T> &input, bool &flag) {

  Ops *ops = NULL;
  unsigned i, no_of_dimensions;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  if (no_of_dimensions == input.getNoOfDimensions()) {
    for (i = 0; i < no_of_dimensions; i++) {
      if (Tensor<T>::getDimensions()[i] != input.getDimensions()[i]) {
        flag = false;
        break;
      }
    }
    if (flag) {
      ops = new Opsadd;
      Tensor<T> *inputs[2];
      inputs[0] = this;
      inputs[1] = &input;
      ops->initilizeinputs(inputs, (unsigned)2);
    } else {
      std::cout << "Error!" << i
                << "th Dimension does not match with second matrix.\n";
    }
  } else {
    std::cout << "Dimension mismatch, First matrix and second matrix has "
                 "different rank.\n";
  }
  return ops;
}

// template <typename T> Graph Tensor<T>::matmul(Graph &g, Tensor<T> &input) {

//   unsigned i, no_of_dimensions;
//   bool flag = true;

//   no_of_dimensions = Tensor<T>::getNoOfDimensions();

//   if (no_of_dimensions == input.getNoOfDimensions()) {

//     if (this->getDimensions()[0] == input.getDimensions()[1]) {

//       for (i = 2; i < no_of_dimensions; i++) {
//         if (Tensor<T>::getDimensions()[i] != input.getDimensions()[i]) {
//           flag = false;
//           break;
//         }
//       }
//       if (flag) {
//         Ops *ops = new Opsmatmul;
//         Tensor<T> *inputs[2];
//         inputs[0] = this;
//         inputs[1] = &input;
//         ops->initilizeinputs(inputs, (unsigned)2);
//         g.addcomputenode(ops);
//       } else {
//         std::cout << "Error!" << i
//                   << "th Dimension does not match with second matrix.\n";
//         g.setGraphInvalid();
//       }
//     } else {
//       std::cout << "Error! First matrix's row length does not match with "
//                    "second matrix column length.\n";
//       g.setGraphInvalid();
//     }
//   } else {
//     std::cout << "Dimension mismatch, First matrix doesn't have same no of "
//                  "dimension of second matrix.\n";
//     g.setGraphInvalid();
//   }

//   return g;
// }

// template <typename T> Graph Tensor<T>::pow(Graph &g, unsigned power) {

//   Ops *ops = new Opspower;
//   Tensor<T> *inputs[1];
//   inputs[0] = this;
//   ops->initilizeinputs(inputs, power);
//   g.addcomputenode(ops);
//   return g;
// }

// template <typename T> Graph Tensor<T>::mul(Graph &g, Tensor<T> &input) {

//   unsigned i, no_of_dimensions;
//   bool flag = true;

//   no_of_dimensions = Tensor<T>::getNoOfDimensions();

//   if (no_of_dimensions == input.getNoOfDimensions()) {
//     for (i = 0; i < no_of_dimensions; i++) {
//       if (Tensor<T>::getDimensions()[i] != input.getDimensions()[i]) {
//         flag = 0;
//         break;
//       }
//     }
//     if (flag) {
//       Ops *ops = new Opsmul;
//       Tensor<T> *inputs[2];
//       inputs[0] = this;
//       inputs[1] = &input;
//       ops->initilizeinputs(inputs, (unsigned)2);
//       g.addcomputenode(ops);
//     } else {
//       std::cout << "Error! " << i
//                 << "th Dimension does not match with second matrix.\n";
//       g.setGraphInvalid();
//     }
//   } else {
//     std::cout << "Dimension mismatch, First matrix doesn't have same no of "
//                  "dimension of second matrix.\n";
//     g.setGraphInvalid();
//   }

//   return g;
// }

// template <typename T>
// Graph Tensor<T>::reducesum(Graph &g, unsigned n, unsigned *reduction_dims) {

//   Ops *ops = new Opsreducesum;
//   for (int i = 0; i < n; i++)
//     reduction_dims[i] = this->getNoOfDimensions() - reduction_dims[i] - 1;

//   // shorting dimensions using bubble short
//   for (unsigned j = 0; j < n; j++)
//     for (unsigned i = 0; i < n - j - 1; i++)
//       if (reduction_dims[i] < reduction_dims[i + 1]) {
//         unsigned temp = reduction_dims[i];
//         reduction_dims[i] = reduction_dims[i + 1];
//         reduction_dims[i + 1] = temp;
//       }

//   Tensor<T> *input[1];
//   input[0] = this;
//   ops->initilizeinputs(input, n, reduction_dims);
//   g.addcomputenode(ops);
//   return g;
// }

// template <typename T> void Tensor<T>::reducesum(Graph &g, Ops *ops) {

//   unsigned count = 0;
//   unsigned *reduction_dims;
//   unsigned *dims;

//   unsigned *arr_dims = new unsigned[this->getNoOfDimensions()];

//   ptr = ptr_prev = head;

//   while (ptr) {
//     count++;
//     ptr = ptr->next;
//   }

//   reduction_dims = new unsigned[count];
//   ptr = head;

//   for (unsigned i = 0; i < count; i++) {
//     reduction_dims[i] = this->getNoOfDimensions() - ptr->value - 1;
//     ptr = ptr->next;
//     delete[] ptr_prev;
//     ptr_prev = ptr;
//   }
//   // shorting dimensions using bubble short
//   for (unsigned j = 0; j < count; j++)
//     for (unsigned i = 0; i < count - j - 1; i++)
//       if (reduction_dims[i] < reduction_dims[i + 1]) {
//         unsigned temp = reduction_dims[i];
//         reduction_dims[i] = reduction_dims[i + 1];
//         reduction_dims[i + 1] = temp;
//       }

//   Tensor<T> *input[1];
//   input[0] = this;
//   ops->initilizeinputs(input, count, reduction_dims);
//   g.addcomputenode(ops);

//   delete[] reduction_dims;
// }

// template <typename T>
// template <typename first_dim, typename... Args>
// void Tensor<T>::reducesum(Graph &g, Ops *ops, first_dim n, Args... args) {
//   if (n < this->getNoOfDimensions()) {

//     ptr = new struct arg_list;
//     ptr->value = n;

//     if (!head) {

//       head = ptr_prev = ptr;
//       head->next = NULL;
//     } else {
//       ptr->next = NULL;
//       ptr_prev->next = ptr;
//       ptr_prev = ptr;
//     }
//     reducesum(g, ops, args...);
//   } else
//     std::cout << "Fatal error! reduction axis does not belong for the Tensor\n";
// }

// template <typename T>
// template <typename... Args>
// Graph Tensor<T>::reducesum(Graph &g, Args... args) {
//   head = NULL;
//   Ops *ops = new Opsreducesum;
//   reducesum(g, ops, args...);
//   return g;
// }

// template <typename T>
// Graph Tensor<T>::scale(Graph &g, const std::float64_t scale_factor) {
//   Ops *ops = new Opsscale;
//   Tensor<T> *inputs[1];
//   inputs[0] = this;
//   ops->initilizeinputs(inputs, (T)scale_factor);
//   g.addcomputenode(ops);
//   return g;
// }

// template <typename T> Graph Tensor<T>::mean(Graph &g, const unsigned n) {
//   static Tensor<T> temp_reduction;

//   if (n < this->getNoOfDimensions()) {
//     temp_reduction.assign(this->reducesum(g, n));
//     return temp_reduction.scale(g, 1.0f / this->getDimensions()[n]);
//   } else {
//     g.setGraphInvalid();
//     return g;
//   }
// }

// template class Tensor<std::bfloat16_t>;
// template class Tensor<std::float16_t>;
// template class Tensor<std::float32_t>;
template class Tensor<std::float64_t>;
// template class Tensor<int8_t>;
// template class Tensor<int16_t>;
// template class Tensor<int32_t>;
// template class Tensor<int64_t>;
// template class Tensor<uint8_t>;
// template class Tensor<uint16_t>;
// template class Tensor<uint32_t>;
// template class Tensor<uint64_t>;