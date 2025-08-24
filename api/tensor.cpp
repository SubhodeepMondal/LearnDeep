#include "tensor.h"

struct tf::arg_list *arg_head;
struct tf::arg_list *arg_ptr;
struct tf::arg_list *arg_ptr_prev;

tf::graph_manager tf::g_manager;

bool tf::graph_manager::isThereActiveSession() {
  return std::any_of(graph_list.begin(), graph_list.end(),
                     [](graph *g) { return g->isSessionActive; });
}

tf::graph *tf::graph_manager::findActivateSession() {
  auto it = std::find_if(graph_list.begin(), graph_list.end(),
                         [](graph *g) { return g->isSessionActive; });
  return *it;
}

void tf::tensor::tensor_of(double low_limit, double upper_limit) {

  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->initRandData(low_limit,
                                                             upper_limit);
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor::tensor_of(std::float64_t *data) {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->initData(data);
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor::print_data() {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->printData();
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor::print_dimension() {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->printDimensions();
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor::operator=(graph &g) {
  if (g.input_a.dt_type == this->dt_type) {
    switch (this->dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(this->ptr)->assign(g.ops);
      static_cast<Graph *>(g.ptr)->addNode(
          static_cast<Tensor<std::float64_t> *>(this->ptr));
      static_cast<Graph *>(g.ptr)->addEdge(
          g.ops, static_cast<Tensor<std::float64_t> *>(this->ptr));

      break;
    default:
      std::cout << "Invalid data type!";
    }
  }
}

tf::graph &tf::tensor::add(graph &g, tensor &input_b) {

  bool flag = true;
  g.input_a = *this;
  g.input_b = input_b;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->add(
      *(static_cast<Graph *>(g.ptr)),
      *(reinterpret_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);

  return g;
}

tf::graph &tf::tensor::mean(graph &g, unsigned dim) {
  bool flag = true;
  g.input_a = *this;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->mean(
      *(static_cast<Graph *>(g.ptr)), dim, flag);

  return g;
}

tf::graph &tf::tensor::mul(graph &g, tensor &input_b) {
  bool flag = true;
  g.input_a = *this;
  g.input_b = input_b;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->mul(
      *(static_cast<Graph *>(g.ptr)),
      *(reinterpret_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);

  return g;
}

tf::graph &tf::tensor::getReductionGraph(graph &g,
                                         std::vector<unsigned> reduction_dims,
                                         bool &flag) {

  // bool flag = true;
  g.input_a = *this;
  g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->reducesum(
      *(static_cast<Graph *>(g.ptr)), reduction_dims, flag);

  return g;
}

tf::graph &tf::tensor::matmul(graph &g, tensor &input_b) {
  switch (dt_type) {
  case tf_float64: {
    bool flag = true;
    g.input_a = *this;
    g.input_b = input_b;
    g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->matmul(
        *(static_cast<Graph *>(g.ptr)),
        *(reinterpret_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);
    break;
  }
  }
  return g;
}

tf::graph &tf::tensor::scale(graph &g, std::float64_t scaleFactor) {
  switch (dt_type) {
  case tf_float64: {
    bool flag = true;
    g.input_a = *this;
    g.ops = reinterpret_cast<Tensor<std::float64_t> *>(this->ptr)->scale(
        *(static_cast<Graph *>(g.ptr)), scaleFactor, flag);
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

    g.input_a = *this;
    g.ops = static_cast<Tensor<std::float64_t> *>(this->ptr)->power(
        *(static_cast<Graph *>(g.ptr)), exponent, flag);
  } break;
  }

  return g;
}

tf::tensor tf::tensor::eager_matmul(tensor &input_b) {
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
      std::cout << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::eager_add(tensor &input_b) {
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
      std::cout << "Invalid data type!";
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
      std::cout << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::eager_scale(const std::float64_t scaleFactor) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr =
        static_cast<Tensor<std::float64_t> *>(this->ptr)->scale(scaleFactor);
    break;
  default:
    std::cout << "Invalid data type!";
  }

  return output;
}

tf::tensor tf::tensor::eager_pow(const unsigned exponent) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr =
        static_cast<Tensor<std::float64_t> *>(this->ptr)->pow(exponent);
    break;
  default:
    std::cout << "Invalid data type!";
  }

  return output;
}

tf::tensor tf::tensor::eager_mean(const unsigned dim) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->mean(dim);
    break;
  default:
    std::cout << "Invalid data type!";
  }

  return output;
}

tf::tensor
tf::tensor::eager_getReduction(std::vector<unsigned> reduction_dims) {
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

void tf::graph::tf_create_graph() {
  ptr = new Graph();
  g_manager.addGraph(this);
}

void tf::graph::graph_start_recording_session() {
  if (g_manager.isThereActiveSession()) {
    std::cout << "A session is already active. Cannot start a new session.\n";
    return;
  }
  this->isSessionActive = true;
  std::cout << "Starting a new recording session.\n";
}

void tf::graph::graph_end_recording_session() {
  if (g_manager.isThereActiveSession()) {
    std::cout << "Ending the active session.\n";
    this->isSessionActive = false;
    g_manager.removeGraph(this);
  }
}

void tf::graph::graph_execute() { static_cast<Graph *>(this->ptr)->compute(); }

void tf::graph::graph_travarse_data_node() {
  static_cast<Graph *>(this->ptr)->traverse();
}

void tf::graph::graph_clear() {
  if (ptr) {
    delete static_cast<Graph *>(ptr);
    ptr = nullptr;
  }
  g_manager.removeGraph(this);
  isSessionActive = false;
  input_a.ptr = nullptr;
  input_b.ptr = nullptr;
  ops = nullptr;
  std::cout << "Graph cleared and session ended.\n";
}