// Library Headers
#include "MathLibrary.h"
#include "NDynamicArray.h"
#include <core/graph/graph_manager.hpp>
// Eager Mode

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
    std::cout
        << "Two metrix requires same shape to perform matrix subtraction, "
           "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
    return output;
  }
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
    std::cout << "Two metrix requires same shape to perform element wise "
                 "multiplication, "
                 "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::add(Tensor<T> &input, bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  Ops *opsadd = new Opsadd();
  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[2];
  inputs[0] = this;
  inputs[1] = &input;
  opsadd->initializeinputs(inputs);
  opsadd->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(&input);
    g->addNode(opsadd);

    g->addEdge(this, opsadd);
    g->addEdge(&input, opsadd);

    g->addNode(output);
    g->addEdge(opsadd, output);
  } else {
    opsadd->compute();
    delete opsadd;
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::div(Tensor<T> &input, bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  unsigned flag = 1;

  // no_of_dimensions = Tensor<T>::getNoOfDimensions();
  if (flag) {
    Ops *opsdiv = new Opsdiv();
    output =
        new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    Tensor<T> *inputs[2];
    inputs[0] = this;
    inputs[1] = &input;
    opsdiv->initializeinputs(inputs);
    opsdiv->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g && graph_flag) {
      g->addNode(this);
      g->addNode(&input);
      g->addNode(opsdiv);

      g->addEdge(this, opsdiv);
      g->addEdge(&input, opsdiv);

      g->addNode(output);
      g->addEdge(opsdiv, output);
    } else {
      opsdiv->compute();
      delete opsdiv;
    }

  } else {
    std::cout << "Two metrix requires same shape to perform elemenet-wise "
                 "multiplication, "
                 "here matrix A ";
    Tensor<T>::printDimensions();
    std::cout << " and matrix B ";
    input.printDimensions();
    std::cout << " are of differenct shape.\n";
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::greaterThanZero(bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;
  Ops *opsgreaterthanzero = new Opsgreaterthanzero();

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  opsgreaterthanzero->initializeinputs(inputs);
  opsgreaterthanzero->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(opsgreaterthanzero);

    g->addEdge(this, opsgreaterthanzero);

    g->addNode(output);
    g->addEdge(opsgreaterthanzero, output);
  } else {
    opsgreaterthanzero->compute();
    delete opsgreaterthanzero;
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::log(bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;
  Ops *opslog = new Opslog();

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  opslog->initializeinputs(inputs);
  opslog->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(opslog);

    g->addEdge(this, opslog);

    g->addNode(output);
    g->addEdge(opslog, output);
  } else {
    opslog->compute();
    delete opslog;
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::matmul(Tensor<T> &input, bool graph_flag) {
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
        Ops *opsmatmul = new Opsmatmul();
        output = new Tensor<T>(Tensor<T>::getNoOfDimensions(), output_dim,
                               this->getType());
        delete[] output_dim;
        Tensor<T> *inputs[2];
        inputs[0] = this;
        inputs[1] = &input;
        opsmatmul->initializeinputs(inputs);
        opsmatmul->initializeoutput(output);

        Graph *g = GraphManager::instance().getCurrentGraph();
        if (g && graph_flag) {
          g->addNode(this);
          g->addNode(&input);
          g->addNode(opsmatmul);

          g->addEdge(this, opsmatmul);
          g->addEdge(&input, opsmatmul);

          g->addNode(output);
          g->addEdge(opsmatmul, output);
        } else {
          opsmatmul->compute();
          delete opsmatmul;
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

template <typename T>
Tensor<T> *Tensor<T>::mul(Tensor<T> &input, bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  unsigned flag = 1;

  // no_of_dimensions = Tensor<T>::getNoOfDimensions();
  if (flag) {
    Ops *opsmul = new Opsmul();
    output =
        new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
    Tensor<T> *inputs[2];
    inputs[0] = this;
    inputs[1] = &input;
    opsmul->initializeinputs(inputs);
    opsmul->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g && graph_flag) {
      g->addNode(this);
      g->addNode(&input);
      g->addNode(opsmul);

      g->addEdge(this, opsmul);
      g->addEdge(&input, opsmul);

      g->addNode(output);
      g->addEdge(opsmul, output);
    } else {
      opsmul->compute();
      delete opsmul;
    }

  } else {
    std::cout << "Two metrix requires same shape to perform elemenet-wise "
                 "multiplication, "
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
template <typename T>
Tensor<T> *Tensor<T>::reducesum(std::vector<unsigned> axis, bool keep_dims,
                                bool graph_flag) {
  Tensor<T> *output;
  unsigned i, no_of_dimensions, count = 0;
  bool flag;

  std::sort(axis.begin(), axis.end());

  no_of_dimensions = this->getNoOfDimensions();

  for (i = 0; i < axis.size(); i++) {
    if (axis[i] >= no_of_dimensions) {
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
    Ops *opsreducesum = new Opsreducesum(keep_dims);
    Tensor<T> *inputs[1];
    inputs[0] = this;
    opsreducesum->initializeinputs(inputs);
    opsreducesum->initializeReductionDims(axis.size(), axis.data());
    opsreducesum->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g && graph_flag) {
      g->addNode(this);
      g->addNode(opsreducesum);

      g->addEdge(this, opsreducesum);

      g->addNode(output);
      g->addEdge(opsreducesum, output);
    } else {
      opsreducesum->compute();
      delete opsreducesum;
    }
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::scale(const std::float64_t scaleFactor, bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;
  Ops *opsscale = new Opsscale();

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  opsscale->initializeinputs(inputs);
  opsscale->initializeScale(scaleFactor);
  opsscale->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(opsscale);

    g->addEdge(this, opsscale);

    g->addNode(output);
    g->addEdge(opsscale, output);
  } else {
    opsscale->compute();
    delete opsscale;
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::sqrt(bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;
  Ops *opssqrt = new Opssqrt();

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  opssqrt->initializeinputs(inputs);
  opssqrt->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(opssqrt);

    g->addEdge(this, opssqrt);

    g->addNode(output);
    g->addEdge(opssqrt, output);
  } else {
    opssqrt->compute();
    delete opssqrt;
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::sub(Tensor<T> &input, bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;

  unsigned flag = 1;
  Ops *opssub = new Opssub();
  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[2];
  inputs[0] = this;
  inputs[1] = &input;
  opssub->initializeinputs(inputs);
  opssub->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(&input);
    g->addNode(opssub);

    g->addEdge(this, opssub);
    g->addEdge(&input, opssub);

    g->addNode(output);
    g->addEdge(opssub, output);
  } else {
    opssub->compute();
    delete opssub;
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::pow(const unsigned exponent, bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;
  Ops *opspow = new Opspower();

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  if (exponent == 0) {
    output->initData(1);
  } else if (exponent == 1) {
    output->initData(this->getData());
  } else {

    Tensor<T> *inputs[1];
    inputs[0] = this;

    opspow->initializeinputs(inputs);
    opspow->initializeExpoent(exponent);
    opspow->initializeoutput(output);

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g && graph_flag) {
      g->addNode(this);
      g->addNode(opspow);

      g->addEdge(this, opspow);

      g->addNode(output);
      g->addEdge(opspow, output);
    } else {
      opspow->compute();
      delete opspow;
    }
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::relu(bool graph_flag) {
  Tensor<T> *output;
  DataType d_type = tf_float64;
  Ops *opsrelu = new Opsrelu();

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  opsrelu->initializeinputs(inputs);
  opsrelu->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(opsrelu);

    g->addEdge(this, opsrelu);

    g->addNode(output);
    g->addEdge(opsrelu, output);
  } else {
    opsrelu->compute();
    delete opsrelu;
  }
  return output;
}

template <typename T> Tensor<T> *Tensor<T>::sigmoid(bool graph_flag) {

  Tensor<T> *output;
  DataType d_type = tf_float64;
  Ops *opssigmoid = new Opssigmoid();

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  opssigmoid->initializeinputs(inputs);
  opssigmoid->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(opssigmoid);

    g->addEdge(this, opssigmoid);

    g->addNode(output);
    g->addEdge(opssigmoid, output);
  } else {
    opssigmoid->compute();
    delete opssigmoid;
  }
  return output;
}

template <typename T>
Tensor<T> *Tensor<T>::softmax(const unsigned axis, bool graph_flag) {

  Tensor<T> *output;
  Ops *ops = new Opssoftmax;
  DataType d_type = tf_float64;

  output =
      new Tensor<T>(this->getNoOfDimensions(), this->getDimensions(), d_type);
  Tensor<T> *inputs[1];
  inputs[0] = this;
  ops->initializeinputs(inputs);
  ops->initializeAxis(axis);
  ops->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
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
Tensor<T> *Tensor<T>::mean(const unsigned dim, bool graph_flag) {
  Tensor<T> *output;
  Tensor<T> *temp_reducesum;
  DataType d_type = tf_float64;
  Ops *opsreducesum = new Opsreducesum(true);
  Ops *opsscale = new Opsscale();

  // first perform reducesum operation along the specified dimension
  if (this->getNoOfDimensions() - 1 <= 0) {
    unsigned arr[1] = {1};
    temp_reducesum = new Tensor<T>(1, arr, d_type);
  } else {

    temp_reducesum = new Tensor<T>(this->getNoOfDimensions() - 1,
                                   this->getDimensions(), d_type);
  }
  Tensor<T> *inputs[1];
  inputs[0] = this;
  unsigned reduction_dim = dim;
  unsigned dims[1] = {reduction_dim};
  opsreducesum->initializeinputs(inputs);
  opsreducesum->initializeReductionDims(1, dims);
  opsreducesum->initializeoutput(temp_reducesum);

  // then perform scale operation with scale factor = 1/n, n = size of the
  output = new Tensor<T>(temp_reducesum->getNoOfDimensions(),
                         temp_reducesum->getDimensions(), d_type);
  std::float64_t scale_factor = 1.0f / this->getDimensions()[reduction_dim];
  opsscale->initializeinputs(&temp_reducesum);
  opsscale->initializeScale(scale_factor);
  opsscale->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    // Ops reduce
    g->addNode(this);
    g->addNode(opsreducesum);
    g->addEdge(this, opsreducesum);

    g->addNode(temp_reducesum);
    g->addEdge(opsreducesum, temp_reducesum);

    // Ops scale
    g->addNode(temp_reducesum);
    g->addNode(opsscale);
    g->addEdge(temp_reducesum, opsscale);

    g->addNode(output);
    g->addEdge(opsscale, output);
  } else {
    opsreducesum->compute();
    opsscale->compute();
    delete temp_reducesum;
    delete opsreducesum;
    delete opsscale;
  }

  return output;
}

template <typename T> Tensor<T> *Tensor<T>::transpose(bool graph_flag) {
  Tensor<T> *output;
  Ops *opstranspose = new Opstranspose();
  std::vector<unsigned> dims(this->getDimensions(),
                             this->getDimensions() + this->getNoOfDimensions());

  // Swap dimensions
  dims[0] = dims[0] + dims[1];
  dims[1] = dims[0] - dims[1];
  dims[0] = dims[0] - dims[1];

  output = new Tensor<T>(dims.size(), dims.data(), tf_float64);

  Tensor<T> *inputs[1];
  inputs[0] = this;

  opstranspose->initializeinputs(inputs);
  opstranspose->initializeoutput(output);

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g && graph_flag) {
    g->addNode(this);
    g->addNode(opstranspose);

    g->addEdge(this, opstranspose);

    g->addNode(output);
    g->addEdge(opstranspose, output);
  } else {
    opstranspose->compute();
    delete opstranspose;
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