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

template <typename T>
Tensor<T> *Tensor<T>::scale(const std::float64_t scaleFactor) {
  Tensor<T> *output;
  Ops *ops;
  DataType d_type = tf_float64;

  ops = new Opsscale;
  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops->initilizeinputs(inputs, scaleFactor);
  ops->initilizeoutput(output);
  ops->compute();

  delete ops;
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::sqrt() {
  Tensor<T> *output;
  Ops *ops = new Opssqrt;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops->initilizeinputs(inputs, (unsigned)1);
  ops->initilizeoutput(output);
  ops->compute();

  delete ops;
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::sub(Tensor<T> &input) {
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
    ops = new Opssub;
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

template <typename T> Tensor<T> *Tensor<T>::pow(const unsigned exponent) {
  Tensor<T> *output;
  DataType d_type = tf_float64;
  Ops *ops;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  if (exponent == 0) {
    output->initData(1);
  } else if (exponent == 1) {
    output->initData(this->getData());
  } else {
    ops = new Opspower;

    Tensor<T> *inputs[1];
    inputs[0] = this;

    ops->initilizeinputs(inputs, exponent);
    ops->initilizeoutput(output);
    ops->compute();

    delete ops;
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::relu() {
  Tensor<T> *output;
  Ops *ops = new Opsrelu;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops->initilizeinputs(inputs, (unsigned)1);
  ops->initilizeoutput(output);
  ops->compute();

  delete ops;
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::mean(const unsigned dim) {
  Tensor<T> *output;
  Tensor<T> *temp_reducesum;
  Ops *ops;
  DataType d_type = tf_float64;

  // first perform reducesum operation along the specified dimension
  ops = new Opsreducesum;
  temp_reducesum = new Tensor<T>(this->getNoOfDimensions() - 1,
                                 this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  unsigned reduction_dim = (this->getNoOfDimensions() - dim - 1) > 0
                               ? (this->getNoOfDimensions() - dim - 1)
                               : 0;
  unsigned dims[1] = {reduction_dim};
  ops->initilizeinputs(inputs, 1, dims);
  ops->initilizeoutput(temp_reducesum);
  ops->compute();
  delete ops;

  // then perform scale operation with scale factor = 1/n, n = size of the
  ops = new Opsscale;
  output = new Tensor<T>(temp_reducesum->getNoOfDimensions(),
                         temp_reducesum->getDimensions(), d_type);
  std::float64_t scale_factor = 1.0f / this->getDimensions()[reduction_dim];
  ops->initilizeinputs(&temp_reducesum, scale_factor);
  ops->initilizeoutput(output);
  ops->compute();
  delete ops;

  return output;
}

template <typename T>
Ops *Tensor<T>::add(Graph &g, Tensor<T> &input, bool &flag) {
  Ops *ops = nullptr;
  Tensor<T> *inputs[2];
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

      inputs[0] = this;
      inputs[1] = &input;

      ops->initilizeinputs(inputs, (unsigned)2);

      g.addNode(this);
      g.addNode(&input);
      g.addNode(ops);

      g.addEdge(this, ops);
      g.addEdge(&input, ops);
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

template <typename T> Ops *Tensor<T>::mean(Graph &g, unsigned dim, bool &flag) {
  Tensor<T> *temp_reducesum;
  Ops *ops;
  DataType d_type = tf_float64;

  // first perform reducesum operation along the specified dimension
  ops = new Opsreducesum;
  temp_reducesum = new Tensor<T>(this->getNoOfDimensions() - 1,
                                 this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  unsigned reduction_dim = (this->getNoOfDimensions() - dim - 1) > 0
                               ? (this->getNoOfDimensions() - dim - 1)
                               : 0;
  unsigned dims[1] = {reduction_dim};

  // initialize the inputs and add nodes and edges to the graph
  ops->initilizeinputs(inputs, 1, dims);
  g.addNode(this);
  g.addNode(ops);
  g.addEdge(this, ops);

  // initialize the output and add nodes and edges to the graph
  ops->initilizeoutput(temp_reducesum);
  g.addNode(temp_reducesum);
  g.addEdge(ops, temp_reducesum);

  // then perform scale operation with scale factor = 1/n, n = size of the
  // dimension
  ops = new Opsscale;
  std::float64_t scale_factor = 1.0f / this->getDimensions()[reduction_dim];
  ops->initilizeinputs(&temp_reducesum, scale_factor);

  g.addNode(temp_reducesum);
  g.addNode(ops);

  g.addEdge(temp_reducesum, ops);

  return ops;
}

template <typename T>
Ops *Tensor<T>::mul(Graph &g, Tensor<T> &input, bool &flag) {

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

      g.addNode(this);
      g.addNode(&input);
      g.addNode(ops);

      g.addEdge(this, ops);
      g.addEdge(&input, ops);
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

template <typename T>
Ops *Tensor<T>::matmul(Graph &g, Tensor<T> &input, bool &flag) {

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

        g.addNode(this);   // Add the first input node to the graph
        g.addNode(&input); // Add the second input node to the graph
        g.addNode(ops);    // Add the operation node to the graph

        g.addEdge(this, ops); // Create an edge from the first input node to the
                              // operation node
        g.addEdge(&input, ops); // Create an edge from the second input node to
                                // the operation node
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
Ops *Tensor<T>::reducesum(Graph &g, std::vector<unsigned> n, bool &flag) {
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

    g.addNode(this);
    g.addNode(ops);

    g.addEdge(this, ops);
  }

  return ops;
}

template <typename T>
Ops *Tensor<T>::pow(Graph &g, const unsigned exponent, bool &flag) {
  Ops *ops = new Opspower;

  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops->initilizeinputs(inputs, exponent);
  g.addNode(this);
  g.addNode(ops);
  g.addEdge(this, ops);
  return ops;
}

template <typename T> Ops *Tensor<T>::relu(Graph &g, bool &flag) {
  Ops *ops = new Opsrelu;
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops->initilizeinputs(inputs, (unsigned)1);
  g.addNode(this);
  g.addNode(ops);
  g.addEdge(this, ops);
  return ops;
}

template <typename T>
Ops *Tensor<T>::scale(Graph &g, const std::float64_t scaleFactor,
                      [[maybe_unused]] bool &flag) {
  Ops *ops = new Opsscale;
  Tensor<T> *inputs[1];
  inputs[0] = this;

  ops->initilizeinputs(inputs, scaleFactor);

  g.addNode(this);
  g.addNode(ops);

  g.addEdge(this, ops);

  return ops;
}

template <typename T> Ops *Tensor<T>::sqrt(Graph &g, bool &flag) {
  Ops *ops = new Opssqrt;
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops->initilizeinputs(inputs, (unsigned)1);
  g.addNode(this);
  g.addNode(ops);
  g.addEdge(this, ops);
  return ops;
}

template <typename T>
Ops *Tensor<T>::sub(Graph &g, Tensor<T> &input, bool &flag) {
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
      ops = new Opssub;

      Tensor<T> *inputs[2];
      inputs[0] = this;
      inputs[1] = &input;

      ops->initilizeinputs(inputs, (unsigned)2);

      g.addNode(this);
      g.addNode(&input);
      g.addNode(ops);

      g.addEdge(this, ops);
      g.addEdge(&input, ops);
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