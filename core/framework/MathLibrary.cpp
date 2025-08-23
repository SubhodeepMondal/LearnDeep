#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>

template <typename T> Tensor<T> *Tensor<T>::matmul(Tensor<T> &input) {
  Tensor<T> *output;
  Ops *ops;
  unsigned i, j, flag = 1;
  unsigned *output_dim;

  if (Tensor<T>::getNoOfDimensions() == input.getNoOfDimensions()) {
    output_dim = new unsigned[Tensor<T>::getNoOfDimensions()];

    output_dim[0] = input.getDimensions()[0];
    output_dim[1] = Tensor<T>::getDimensions()[1];

    if (this->getDimensions()[0] == input.getDimensions()[1]) {

      for (i = 2; i < Tensor<T>::getNoOfDimensions(); i++) {
        output_dim[i] = Tensor<T>::getDimensions()[i];
        if (Tensor<T>::getDimensions()[i] != input.getDimensions()[i]) {
          flag = 0;
          break;
        }
      }
      if (flag) {
        ops = new Opsmatmul;
        output = new Tensor<T>(Tensor<T>::getNoOfDimensions(), output_dim,
                               this->getType());
        Tensor<T> *inputs[2];
        inputs[0] = this;
        inputs[1] = &input;
        ops->initilizeinputs(inputs, (unsigned)2);
        ops->initilizeoutput(output);
        ops->compute();

        delete ops;
        delete[] output_dim;
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

  delete[] arr;

  return output;
}

template <typename T> Tensor<T> *Tensor<T>::operator*(Tensor<T> &input) {
  Tensor<T> *output;
  Ops *ops;
  DataType d_type = tf_float64;

  unsigned flag = 1;

  // no_of_dimensions = Tensor<T>::getNoOfDimensions();

  for (int i = 0; i < this->getNoOfDimensions(); i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    ops = new Opsmul;
    output =
        new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    Tensor<T> *inputs[2];
    inputs[0] = this;
    inputs[1] = &input;
    ops->initilizeinputs(inputs, (unsigned)2);
    ops->initilizeoutput(output);
    ops->compute();

    delete ops;
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

template <typename T> Tensor<T> *Tensor<T>::mul(Tensor<T> &input) {
  Tensor<T> *output;
  Ops *ops;
  DataType d_type = tf_float64;

  unsigned flag = 1;

  // no_of_dimensions = Tensor<T>::getNoOfDimensions();

  for (int i = 0; i < this->getNoOfDimensions(); i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    ops = new Opsmul;
    output =
        new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    Tensor<T> *inputs[2];
    inputs[0] = this;
    inputs[1] = &input;
    ops->initilizeinputs(inputs, (unsigned)2);
    ops->initilizeoutput(output);
    ops->compute();

    delete ops;
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

template <typename T> Tensor<T> *Tensor<T>::add(Tensor<T> &input) {
  Tensor<T> *output;
  Ops *ops;
  DataType d_type = tf_float64;

  unsigned flag = 1;


  for (int i = 0; i < this->getNoOfDimensions(); i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    ops = new Opsadd;
    output =
        new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    Tensor<T> *inputs[2];
    inputs[0] = this;
    inputs[1] = &input;
    ops->initilizeinputs(inputs, (unsigned)2);
    ops->initilizeoutput(output);
    ops->compute();

    delete ops;
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

template <typename T> Tensor<T> *Tensor<T>::reducesum(std::vector<unsigned> n) {
  Tensor<T> *output;
  Ops *ops = NULL;
  unsigned i, no_of_dimensions, count = 0;
  bool flag = true;

  std::sort(n.begin(), n.end());

  no_of_dimensions = this->getNoOfDimensions();

  for (i = 0; i < n.size(); i++) {
    if (n[i] >= no_of_dimensions) {
      flag = false;
      std::cout
          << "Fatal error! reduction axis does not belong for the Tensor\n";
      return nullptr;
    }
    count++;
  }

  if (count > 0) {
    ops = new Opsreducesum;
    output = new Tensor<T>(this->getNoOfDimensions() - count,
                           this->getDimensions(), this->getType());
    Tensor<T> *inputs[1];
    inputs[0] = this;
    ops->initilizeinputs(inputs, n.size(), n.data());
    ops->initilizeoutput(output);
    ops->compute();
  }

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

template <typename T> Ops *Tensor<T>::mul(Tensor<T> &input, bool &flag) {

  Ops *ops = NULL;
  unsigned i, no_of_dimensions;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  if (no_of_dimensions == input.getNoOfDimensions()) {
    for (i = 0; i < no_of_dimensions; i++) {
      if (Tensor<T>::getDimensions()[i] != input.getDimensions()[i]) {
        flag = 0;
        break;
      }
    }
    if (flag) {
      ops = new Opsmul;
      Tensor<T> *inputs[2];
      inputs[0] = this;
      inputs[1] = &input;
      ops->initilizeinputs(inputs, (unsigned)2);
    } else {
      std::cout << "Error! " << i
                << "th Dimension does not match with second matrix.\n";
    }
  } else {
    std::cout << "Dimension mismatch, First matrix doesn't have same no of "
                 "dimension of second matrix.\n";
  }

  return ops;
}

template <typename T> Ops *Tensor<T>::matmul(Tensor<T> &input, bool &flag) {

  Ops *ops = NULL;
  unsigned i, no_of_dimensions;

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  if (no_of_dimensions == input.getNoOfDimensions()) {

    if (this->getDimensions()[0] == input.getDimensions()[1]) {

      for (i = 2; i < no_of_dimensions; i++) {
        if (Tensor<T>::getDimensions()[i] != input.getDimensions()[i]) {
          flag = false;
          break;
        }
      }
      if (flag) {
        ops = new Opsmatmul;
        Tensor<T> *inputs[2];
        inputs[0] = this;
        inputs[1] = &input;
        ops->initilizeinputs(inputs, (unsigned)2);
      } else {
        std::cout << "Error!" << i
                  << "th Dimension does not match with second matrix.\n";
      }
    } else {
      std::cout << "Error! First matrix's row length does not match with "
                   "second matrix column length.\n";
    }
  } else {
    std::cout << "Dimension mismatch, First matrix doesn't have same no of "
                 "dimension of second matrix.\n";
  }

  return ops;
}

template <typename T>
Ops *Tensor<T>::reducesum(std::vector<unsigned> n, bool &flag) {
  Ops *ops = NULL;
  unsigned i, no_of_dimensions, count = 0;

  std::sort(n.begin(), n.end());

  no_of_dimensions = Tensor<T>::getNoOfDimensions();

  for (i = 0; i < n.size(); i++) {
    if (n[i] >= no_of_dimensions) {
      flag = false;
      std::cout
          << "Fatal error! reduction axis does not belong for the Tensor\n";
      return ops;
    }
    count++;
  }

  if (count > 0) {
    ops = new Opsreducesum;
    Tensor<T> *inputs[1];
    inputs[0] = this;
    ops->initilizeinputs(inputs, n.size(), n.data());
  }

  return ops;
}

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