#include "MathLibrary.h"
#include "NDynamicArray.h"
#include <graph/graph_manager.hpp>
// Eager Mode
template <typename T>
Tensor<T> *Tensor<T>::add(Tensor<T> &input, std::span<Ops *> ops) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  unsigned flag = 1;

  for (int i = 0; i < this->getNoOfDimensions(); i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    output =
        new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    Tensor<T> *inputs[2];
    inputs[0] = this;
    inputs[1] = &input;
    ops[0]->initializeinputs(inputs);
    ops[0]->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g) {
      g->addNode(this);
      g->addNode(&input);
      g->addNode(ops[0]);

      g->addEdge(this, ops[0]);
      g->addEdge(&input, ops[0]);

      g->addNode(output);
      g->addEdge(ops[0], output);
    } else {
      ops[0]->compute();
    }

  } else {
    std::cout << "Two metrix requires same shape to perform matrix addition, "
                 "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::matmul(Tensor<T> &input, std::span<Ops *> ops) {
  Tensor<T> *output;
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
        output = new Tensor<T>(Tensor<T>::getNoOfDimensions(), output_dim,
                               this->getType());
        delete[] output_dim;
        Tensor<T> *inputs[2];
        inputs[0] = this;
        inputs[1] = &input;
        ops[0]->initializeinputs(inputs);
        ops[0]->initializeoutput(output);

        Graph *g = GraphManager::instance().getCurrentGraph();
        if (g) {
          g->addNode(this);
          g->addNode(&input);
          g->addNode(ops[0]);

          g->addEdge(this, ops[0]);
          g->addEdge(&input, ops[0]);

          g->addNode(output);
          g->addEdge(ops[0], output);
        } else {
          ops[0]->compute();
        }
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
    ops->initializeinputs(inputs);
    ops->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g) {
      g->addNode(this);
      g->addNode(&input);
      g->addNode(ops);

      g->addEdge(this, ops);
      g->addEdge(&input, ops);

      g->addNode(output);
      g->addEdge(ops, output);
    } else {
      ops->compute();
      delete ops;
    }
  } else {
    std::cout << "Two metrix requires same shape to perform matrix addition, "
                 "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::mul(Tensor<T> &input, std::span<Ops *> ops) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  unsigned flag = 1;

  // no_of_dimensions = Tensor<T>::getNoOfDimensions();

  for (int i = 0; i < this->getNoOfDimensions(); i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    output =
        new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    Tensor<T> *inputs[2];
    inputs[0] = this;
    inputs[1] = &input;
    ops[0]->initializeinputs(inputs);
    ops[0]->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g) {
      g->addNode(this);
      g->addNode(&input);
      g->addNode(ops[0]);

      g->addEdge(this, ops[0]);
      g->addEdge(&input, ops[0]);

      g->addNode(output);
      g->addEdge(ops[0], output);
    } else {
      ops[0]->compute();
    }

  } else {
    std::cout << "Two metrix requires same shape to perform matrix addition, "
                 "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
  }
  return output;
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
    } else {
      for (int i = 2; i < no_of_dimensions; i++)
        for (int j = 0; j < this->getDimensions()[i]; j++) {
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
    } else {
      for (int i = 2; i < no_of_dimensions; i++)
        for (int j = 0; j < this->getDimensions()[i]; j++) {
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

template <typename T>
Tensor<T> *Tensor<T>::reducesum(std::vector<unsigned> n, std::span<Ops *> ops) {
  Tensor<T> *output;
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
    output = new Tensor<T>(this->getNoOfDimensions() - count,
                           this->getDimensions(), this->getType());
    Tensor<T> *inputs[1];
    inputs[0] = this;
    ops[0]->initializeinputs(inputs);
    ops[0]->initializeReductionDims(n.size(), n.data());
    ops[0]->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g) {
      g->addNode(this);
      g->addNode(ops[0]);

      g->addEdge(this, ops[0]);

      g->addNode(output);
      g->addEdge(ops[0], output);
    } else {
      ops[0]->compute();
    }
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::scale(const std::float64_t scaleFactor,
                            std::span<Ops *> ops) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops[0]->initializeinputs(inputs);
  ops[0]->initializeScale(scaleFactor);
  ops[0]->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    g->addNode(this);
    g->addNode(ops[0]);

    g->addEdge(this, ops[0]);

    g->addNode(output);
    g->addEdge(ops[0], output);
  } else {
    ops[0]->compute();
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::sqrt(std::span<Ops *> ops) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops[0]->initializeinputs(inputs);
  ops[0]->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    g->addNode(this);
    g->addNode(ops[0]);

    g->addEdge(this, ops[0]);

    g->addNode(output);
    g->addEdge(ops[0], output);
  } else {
    ops[0]->compute();
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::sub(Tensor<T> &input, std::span<Ops *> ops) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  unsigned flag = 1;

  for (int i = 0; i < this->getNoOfDimensions(); i++)
    if (this->getDimensions()[i] != input.getDimensions()[i]) {
      flag = 0;
      break;
    }
  if (flag) {
    output =
        new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    Tensor<T> *inputs[2];
    inputs[0] = this;
    inputs[1] = &input;
    ops[0]->initializeinputs(inputs);
    ops[0]->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g) {
      g->addNode(this);
      g->addNode(&input);
      g->addNode(ops[0]);

      g->addEdge(this, ops[0]);
      g->addEdge(&input, ops[0]);

      g->addNode(output);
      g->addEdge(ops[0], output);
    } else {
      ops[0]->compute();
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

template <typename T>
Tensor<T> *Tensor<T>::pow(const unsigned exponent, std::span<Ops *> ops) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  if (exponent == 0) {
    output->initData(1);
  } else if (exponent == 1) {
    output->initData(this->getData());
  } else {

    Tensor<T> *inputs[1];
    inputs[0] = this;

    ops[0]->initializeinputs(inputs);
    ops[0]->initializeExpoent(exponent);
    ops[0]->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g) {
      g->addNode(this);
      g->addNode(ops[0]);

      g->addEdge(this, ops[0]);

      g->addNode(output);
      g->addEdge(ops[0], output);
    } else {
      ops[0]->compute();
    }
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::relu(std::span<Ops *> ops) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops[0]->initializeinputs(inputs);
  ops[0]->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    g->addNode(this);
    g->addNode(ops[0]);

    g->addEdge(this, ops[0]);

    g->addNode(output);
    g->addEdge(ops[0], output);
  } else {
    ops[0]->compute();
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::sigmoid(std::span<Ops *> ops) {

  Tensor<T> *output;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops[0]->initializeinputs(inputs);
  ops[0]->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    g->addNode(this);
    g->addNode(ops[0]);

    g->addEdge(this, ops[0]);

    g->addNode(output);
    g->addEdge(ops[0], output);
  } else {
    ops[0]->compute();
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::softmax(const unsigned axis) {

  Tensor<T> *output;
  Ops *ops = new Opssoftmax;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops->initializeinputs(inputs);
  ops->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    g->addNode(this);
    g->addNode(ops);

    g->addEdge(this, ops);

    g->addNode(output);
    g->addEdge(ops, output);
  } else {
    ops->compute();
    delete ops;
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::mean(const unsigned dim, std::span<Ops *> ops) {
  Tensor<T> *output;
  Tensor<T> *temp_reducesum;
  DataType d_type = tf_float64;

  // first perform reducesum operation along the specified dimension
  temp_reducesum = new Tensor<T>(this->getNoOfDimensions() - 1,
                                 this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  unsigned reduction_dim = (this->getNoOfDimensions() - dim - 1) > 0
                               ? (this->getNoOfDimensions() - dim - 1)
                               : 0;
  unsigned dims[1] = {reduction_dim};
  ops[0]->initializeinputs(inputs);
  ops[0]->initializeReductionDims(1, dims);
  ops[0]->initializeoutput(temp_reducesum);

  // then perform scale operation with scale factor = 1/n, n = size of the
  output = new Tensor<T>(temp_reducesum->getNoOfDimensions(),
                         temp_reducesum->getDimensions(), d_type);
  std::float64_t scale_factor = 1.0f / this->getDimensions()[reduction_dim];
  ops[1]->initializeinputs(&temp_reducesum);
  ops[1]->initializeScale(scale_factor);
  ops[1]->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    // Ops reduce
    g->addNode(this);
    g->addNode(ops[0]);
    g->addEdge(this, ops[0]);

    g->addNode(temp_reducesum);
    g->addEdge(ops[0], temp_reducesum);

    // Ops scale
    g->addNode(temp_reducesum);
    g->addNode(ops[1]);
    g->addEdge(temp_reducesum, ops[1]);

    g->addNode(output);
    g->addEdge(ops[1], output);
  } else {
    ops[0]->compute();
    ops[1]->compute();
    delete temp_reducesum;
  }

  return output;
}

template <typename T> Tensor<T> *Tensor<T>::transpose(std::span<Ops *> ops) {
  Tensor<T> *output;
  std::vector<unsigned> dims(this->getDimensions(),
                             this->getDimensions() + this->getNoOfDimensions());

  // Swap dimensions
  dims[0] = dims[0] + dims[1];
  dims[1] = dims[0] - dims[1];
  dims[0] = dims[0] - dims[1];

  output = new Tensor<T>(dims.size(), dims.data(), tf_float64);

  Tensor<T> *inputs[1];
  inputs[0] = this;

  ops[0]->initializeinputs(inputs);
  ops[0]->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    g->addNode(this);
    g->addNode(ops[0]);

    g->addEdge(this, ops[0]);

    g->addNode(output);
    g->addEdge(ops[0], output);
  } else {
    ops[0]->compute();
  }

  return output;
}
// End Eager Mode

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