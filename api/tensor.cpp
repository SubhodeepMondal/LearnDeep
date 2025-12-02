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

// ---------------- Eager Mode ---------------
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

tf::tensor tf::tensor::mul(tensor &input_b) {
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
// ------------- End Eager Mode -------------

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

// -------------- Graph Context ------------
tf::graph_context::graph_context() { this->graph_ctx = new GraphContext(); }

tf::graph_context::~graph_context() {

  std::vector<void *> data_nodes =
      static_cast<GraphContext *>(this->graph_ctx)->get_data_nodes();

  for (auto tensor_node : tensor_nodes) {
    for (auto node : data_nodes) {
      if (node == tensor_node->ptr) {
        tensor_node->isNodeCleared = true;
        tensor_node->ptr = nullptr;
        break;
      }
    }
  }

  static_cast<GraphContext *>(this->graph_ctx)->~GraphContext();
}

tf::tensor tf::graph_context::get_gradient(const tensor &a) {
  tensor output;
  output.dt_type = a.dt_type;
  Tensor<std::float64_t> *temp_ptr =
      static_cast<GraphContext *>(this->graph_ctx)
          ->graph_get_gradient(
              reinterpret_cast<Tensor<std::float64_t> *>(a.ptr));

  if (temp_ptr)
    output.ptr = temp_ptr;

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