// Library Headers
#include "tensor.h"
#include <absl/log/log.h>

// --- Default Constructor
tf::tensor::tensor() : ptr(NULL) { tensor_nodes.push_back(this); }

tf::tensor::tensor(DataType dt_type, Tensor<std::float64_t> *ptr) {
  if (ptr) {
    this->ptr = (void *)ptr;
    this->dt_type = dt_type;
  }

  bool flag = true;
  for (tensor *tensor_node : tensor_nodes)
    if (tensor_node->ptr == this->ptr) {
      flag = false;
      break;
    }
  if (flag)
    tensor_nodes.push_back(this);
}

// --- Copy constructor
tf::tensor::tensor(const tensor &other) {
  dt_type = other.dt_type;
  if (other.ptr) {
    auto *src = static_cast<Tensor<std::float64_t> *>(other.ptr);
    ptr = new Tensor<std::float64_t>(*src); // deep copy
  }

  bool flag = true;
  for (tensor *tensor_node : tensor_nodes)
    if (tensor_node->ptr == this->ptr) {
      flag = false;
      break;
    }
  if (flag)
    tensor_nodes.push_back(this);
}

// --- Copy assignment
tf::tensor &tf::tensor::operator=(const tensor &other) {
  if (this != &other) {
    if (other.ptr) {
      if (this->ptr) {
        delete static_cast<Tensor<std::float64_t> *>(this->ptr);
        this->ptr = nullptr;
      }
      this->ptr = new Tensor<std::float64_t>(
          *static_cast<Tensor<std::float64_t> *>(other.ptr));
      this->dt_type = other.dt_type;
    } else {
      ptr = nullptr;
    }
  }
  return *this;
}

// --- Move constructor
tf::tensor::tensor(tensor &&other) noexcept {
  dt_type = other.dt_type;
  ptr = other.ptr;
  other.ptr = nullptr;
  tensor_nodes.push_back(this);
}

// --- Move assignment
tf::tensor &tf::tensor::operator=(tensor &&other) noexcept {
  if (this != &other) {
    if (this->ptr)
      delete static_cast<Tensor<std::float64_t> *>(this->ptr);
    this->dt_type = other.dt_type;
    this->ptr = other.ptr;
    other.ptr = nullptr;
  }

  // for (auto t : tensor_nodes)
  //   std::cout << t << ", ";
  // std::cout << "\n";
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g)
    std::erase(tensor_nodes, this);
  // for (auto t : tensor_nodes)
  //   std::cout << t << ", ";
  // std::cout << "\n";

  return *this;
}

void tf::tensor::assign_pointer(std::vector<unsigned> dimensions) {

  switch (this->dt_type) {
  case tf_float64:
    this->ptr = new Tensor<std::float64_t>(dimensions.size(), dimensions.data(),
                                           this->dt_type);
    break;
  default:
    ptr = nullptr;
  }
}

// --- Destructor
tf::tensor::~tensor() {

  if (tensor_nodes.end() !=
      std::find(tensor_nodes.begin(), tensor_nodes.end(), this)) {
    if (this->ptr) {
      delete static_cast<Tensor<std::float64_t> *>(this->ptr);
      this->ptr = NULL;
    }
    if (opsPtr.size()) {
      for (void *opsptr : this->opsPtr)
        delete static_cast<Ops *>(opsptr);
      opsPtr.clear();
    }
    std::erase(tensor_nodes, this);
  }
}

unsigned tf::tensor::getNoOfDimensions() {
  return static_cast<Tensor<std::float64_t> *>(ptr)->getNoOfDimensions();
}

const unsigned *tf::tensor::getDimensions() {
  return static_cast<Tensor<std::float64_t> *>(ptr)->getDimensions();
}

unsigned tf::tensor::getNoOfElem() {
  return static_cast<Tensor<std::float64_t> *>(ptr)->getNoOfElem();
}

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

// ---------------- Eager Mode ---------------
tf::tensor tf::tensor::matmul(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    opsPtr.push_back(new Opsmatmul);
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->matmul(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)),
          std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &input_b);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::add(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    opsPtr.push_back(new Opsadd);
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->add(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)),
          std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &input_b);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::operator+(tensor &input_b) {
  tensor output;
  if (this->dt_type == input_b.dt_type) {
    opsPtr.push_back(new Opsadd);
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->add(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)),
          std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &input_b);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::operator*(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    opsPtr.push_back(new Opsmul);
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->mul(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)),
          std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &input_b);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::sigmoid() {
  tensor output;
  opsPtr.push_back(new Opssigmoid);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->sigmoid(
        std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::scale(const std::float64_t scaleFactor) {
  tensor output;

  opsPtr.push_back(new Opsscale);
  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->scale(
        scaleFactor, std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::sqrt() {
  tensor output;

  opsPtr.push_back(new Opssqrt);
  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->sqrt(
        std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::sub(tensor &input_b) {
  tensor output;
  opsPtr.push_back(new Opssub);

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->sub(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)),
          std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &input_b);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::transpose() {
  tensor output;
  opsPtr.push_back(new Opstranspose);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->transpose(
        std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::pow(const unsigned exponent) {
  tensor output;
  opsPtr.push_back(new Opspower);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->pow(
        exponent, std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::relu() {
  tensor output;
  opsPtr.push_back(new Opsrelu);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->relu(
        std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::mean(const unsigned dim) {
  tensor output;
  opsPtr.push_back(new Opsreducesum);
  opsPtr.push_back(new Opsscale);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->mean(
        dim, std::span(opsPtr).subspan(opsPtr.size() - 2));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  // for (auto t : tensor_nodes)
  //   std::cout << t << ", ";
  // std::cout << "\n";
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &output);
  }
  // for (auto t : tensor_nodes)
  //   std::cout << t << ", ";
  // std::cout << "\n";
  return output;
}

tf::tensor tf::tensor::mul(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    opsPtr.push_back(new Opsmul);
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->mul(
          *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)),
          std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &input_b);
    std::erase(tensor_nodes, &output);
  }
  return output;
}

tf::tensor tf::tensor::getReduction(std::vector<unsigned> reduction_dims) {
  tensor output;
  opsPtr.push_back(new Opsreducesum);
  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = static_cast<Tensor<std::float64_t> *>(this->ptr)->reducesum(
        reduction_dims, std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;

  default:
    break;
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this);
    std::erase(tensor_nodes, &output);
  }

  return output;
}
// ------------- End Eager Mode -------------

void tf::tensor::gradient_required(bool is_grad_required) {
  if (ptr) {
    static_cast<Tensor<std::float64_t> *>(this->ptr)->gradientRequired(
        is_grad_required);
  }
}

// -------------- Graph Context ------------
tf::graph_context::graph_context() { this->graph_ctx = new GraphContext(); }

tf::graph_context::~graph_context() { delete graph_ctx; }

tf::tensor tf::graph_context::get_gradient(const tensor &a) {
  Tensor<std::float64_t> *temp_ptr =
      static_cast<GraphContext *>(this->graph_ctx)
          ->graph_get_gradient(
              reinterpret_cast<Tensor<std::float64_t> *>(a.ptr));

  tensor output(a.dt_type, temp_ptr);
  std::erase(tensor_nodes, &output);

  return output;
}

void tf::graph_context::run() {
  static_cast<GraphContext *>(this->graph_ctx)->run();
}

void tf::graph_context::initialize_gradient() {
  static_cast<GraphContext *>(this->graph_ctx)->graph_initiize_gradient();
}

void tf::graph_context::compute_gradient() {
  static_cast<GraphContext *>(this->graph_ctx)->graph_compute_gradeint();
}
// -------------- Graph Context ------------