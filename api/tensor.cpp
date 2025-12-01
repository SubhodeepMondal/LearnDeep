#include "tensor.h"
#include <absl/log/log.h>
#include <graph/graph_manager.hpp>

void tf::tensor::assign_ptr() { tensor_nodes.push_back(this); }

void tf::tensor::tensor_of(double low_limit, double upper_limit) {

  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->initRandData(low_limit,
                                                             upper_limit);
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
}

void tf::tensor::tensor_of(std::float64_t *data) {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->initData(data);
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
}

void tf::tensor::print_data() {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->printData();
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
}

void tf::tensor::print_dimension() {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->printDimensions();
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
}

void tf::tensor::operator=(graph &g) {
  if (g.input_a->dt_type == this->dt_type) {
    switch (this->dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(this->ptr)->assign(g.ops);
      static_cast<Graph *>(g.ptr)->addNode(
          static_cast<Tensor<std::float64_t> *>(this->ptr));
      static_cast<Graph *>(g.ptr)->addEdge(
          g.ops, static_cast<Tensor<std::float64_t> *>(this->ptr));
      g.output = this;
      // g.input_a->ptr = nullptr;
      // g.input_b->ptr = nullptr;
      // g.ops = nullptr;

      break;
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
}

tf::graph &tf::tensor::add(graph &g, tensor &input_b) {

  bool flag = true;
  g.input_a = this;
  g.input_b = &input_b;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->add(
      *(static_cast<Graph *>(g.ptr)),
      *(reinterpret_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);

  return g;
}

tf::graph &tf::tensor::mean(graph &g, unsigned dim) {
  bool flag = true;
  g.input_a = this;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->mean(
      *(static_cast<Graph *>(g.ptr)), dim, flag);

  return g;
}

tf::graph &tf::tensor::mul(graph &g, tensor &input_b) {
  bool flag = true;
  g.input_a = this;
  g.input_b = &input_b;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->mul(
      *(static_cast<Graph *>(g.ptr)),
      *(reinterpret_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);

  return g;
}

tf::graph &tf::tensor::getReductionGraph(graph &g,
                                         std::vector<unsigned> reduction_dims,
                                         bool &flag) {
  g.input_a = this;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->reducesum(
      *(static_cast<Graph *>(g.ptr)), reduction_dims, flag);

  return g;
}

tf::graph &tf::tensor::matmul(graph &g, tensor &input_b) {
  switch (dt_type) {
  case tf_float64: {
    bool flag = true;
    g.input_a = this;
    g.input_b = &input_b;
    g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->matmul(
        *(static_cast<Graph *>(g.ptr)),
        *(reinterpret_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);
    break;
  }
  }
  return g;
}

tf::graph &tf::tensor::pow(graph &g, unsigned exponent) {
  // graph *g = nullptr;
  switch (dt_type) {
  case tf_float64: {
    bool flag = true;

    g.input_a = this;
    g.ops = static_cast<Tensor<std::float64_t> *>(this->ptr)->pow(
        *(static_cast<Graph *>(g.ptr)), exponent, flag);
  } break;
  }

  return g;
}

tf::graph &tf::tensor::relu(graph &g) {
  switch (dt_type) {
  case tf_float64: {
    bool flag = true;
    g.input_a = this;
    g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->relu(
        *(static_cast<Graph *>(g.ptr)), flag);
    break;
  }
  }
  return g;
}

tf::graph &tf::tensor::sigmoid(graph &g) {
  switch (dt_type) {
  case tf_float64: {
    bool flag = true;
    g.input_a = this;
    g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->sigmoid(
        *(static_cast<Graph *>(g.ptr)), flag);
    break;
  }
  }
  return g;
}

tf::graph &tf::tensor::scale(graph &g, std::float64_t scaleFactor) {
  switch (dt_type) {
  case tf_float64: {
    bool flag = true;
    g.input_a = this;
    g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->scale(
        *(static_cast<Graph *>(g.ptr)), scaleFactor, flag);
    break;
  }
  }
  return g;
}

tf::graph &tf::tensor::sqrt(graph &g) {
  switch (dt_type) {
  case tf_float64: {
    bool flag = true;
    g.input_a = this;
    g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->sqrt(
        *(static_cast<Graph *>(g.ptr)), flag);
    break;
  }
  }
  return g;
}

tf::graph &tf::tensor::sub(graph &g, tensor &input_b) {
  bool flag = true;
  g.input_a = this;
  g.input_b = &input_b;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->sub(
      *(static_cast<Graph *>(g.ptr)),
      *(reinterpret_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);

  return g;
}

tf::graph &tf::tensor::transpose(graph &g) {
  switch (dt_type) {
  case tf_float64: {
    g.input_a = this;
    g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->transpose(
        *(static_cast<Graph *>(g.ptr)));
    break;
  }
  }
  return g;
}

tf::tensor tf::tensor::matmul(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->matmul(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::add(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->add(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::operator+(tensor &input_b) {
  tensor output;
  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->add(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::operator*(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->mul(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::sigmoid() {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->sigmoid();
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }

  return output;
}

tf::tensor tf::tensor::scale(const std::float64_t scaleFactor) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr =
        static_cast<Tensor<std::float64_t> *>(this->ptr)->scale(scaleFactor);
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }

  return output;
}

tf::tensor tf::tensor::sqrt() {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->sqrt();
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }

  return output;
}

tf::tensor tf::tensor::sub(tensor &input_b) {
  tensor output;
  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->sub(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::transpose() {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->transpose();
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  return output;
}

tf::tensor tf::tensor::pow(const unsigned exponent) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr =
        static_cast<Tensor<std::float64_t> *>(this->ptr)->pow(exponent);
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }

  return output;
}

tf::tensor tf::tensor::relu() {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->relu();
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }

  return output;
}

tf::tensor tf::tensor::mean(const unsigned dim) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->mean(dim);
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }

  return output;
}

tf::tensor tf::tensor::getReduction(std::vector<unsigned> reduction_dims) {
  tensor output;
  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->reducesum(
        reduction_dims);
    break;

  default:
    break;
  }

  return output;
}

void tf::tensor::gradient_required(bool is_grad_required) {
  if (ptr) {
    static_cast<Tensor<std::float64_t> *>(this->ptr)->gradientRequired(
        is_grad_required);
  }
}

void tf::tensor::destory() {
  tensor_nodes.erase(
      std::remove(tensor_nodes.begin(), tensor_nodes.end(), this),
      tensor_nodes.end());
  if (this->ptr) {
    delete static_cast<Tensor<std::float64_t> *>(this->ptr);
    ptr = nullptr;
  }
}

void tf::tensor::eraseRecord() {
  tensor_nodes.erase(
      std::remove(tensor_nodes.begin(), tensor_nodes.end(), this),
      tensor_nodes.end());
}

void tf::graph::tf_create_graph() { ptr = new Graph(); }

void tf::graph::graph_execute() { static_cast<Graph *>(this->ptr)->compute(); }

tf::tensor tf::graph::graph_get_gradient(const tensor &a) {
  tensor output;
  output.dt_type = a.dt_type;
  Tensor<std::float64_t> *temp_ptr =
      static_cast<Graph *>(this->ptr)->getGradientTensor(
          reinterpret_cast<Tensor<std::float64_t> *>(a.ptr));

  if (temp_ptr)
    output.ptr = temp_ptr;

  // static_cast<Tensor<std::float64_t> *>(output.ptr)->printData();
  return output;
}

void tf::graph::graph_travarse_data_node() {
  static_cast<Graph *>(this->ptr)->traverse();
}

void tf::graph::graph_clear() {

  std::vector<void *> data_nodes =
      static_cast<Graph *>(this->ptr)->getDataNodes();

  for (auto tensor_node : tensor_nodes) {
    for (auto node : data_nodes) {
      if (node == tensor_node->ptr) {
        tensor_node->isNodeCleared = true;
        tensor_node->ptr = nullptr;
        break;
      }
    }
  }

  static_cast<Graph *>(this->ptr)->release_resources();
  if (ptr) {
    delete static_cast<Graph *>(this->ptr);
    this->ptr = nullptr;
  }

  tensor_nodes.clear();
}

void tf::graph::graph_initialize_gradient() {
  static_cast<Graph *>(this->ptr)->createGradientGraph();
}

void tf::graph::graph_compute_gradient() {
  static_cast<Graph *>(this->ptr)->computeGradient();
}

void tf::graph::graph_traverse_gradient() {
  static_cast<Graph *>(this->ptr)->traverseGradientGraph();
}