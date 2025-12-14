// Library Headers
#include "tensor.h"
#include <absl/log/log.h>

// --- Default Constructor
tf::tensor::tensor() : ptr(NULL) {}

tf::tensor::tensor(DataType dt_type, Tensor<std::float64_t> *ptr) {
  if (ptr) {
    this->ptr = ptr;
    this->dt_type = dt_type;
  }
  // for (Tensor<std::float64_t> *tensor_node : tensor_nodes)
  //   if (tensor_node->ptr == this->ptr) {
  //     break;
  //   }
}

// --- Copy constructor
tf::tensor::tensor(const tensor &other) {
  dt_type = other.dt_type;
  if (other.ptr) {
    ptr = other.ptr;
  }

  // bool flag = true;
  // for (tensor *tensor_node : tensor_nodes)
  //   if (tensor_node->ptr == this->ptr) {
  //     flag = false;
  //     break;
  //   }
}

// --- Copy assignment
tf::tensor &tf::tensor::operator=(const tensor &other) {
  if (this != &other) {
    if (other.ptr) {
      if (this->ptr) {
        delete this->ptr;
        this->ptr = nullptr;
      }
      this->ptr = other.ptr; // new Tensor<std::float64_t>(*other.ptr);
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
}

// --- Move assignment
tf::tensor &tf::tensor::operator=(tensor &&other) noexcept {
  if (this != &other) {
    if (this->ptr)
      delete this->ptr;
    this->dt_type = other.dt_type;
    this->ptr = other.ptr;
    other.ptr = nullptr;
  }
  tensor_nodes.push_back(this->ptr);
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g)
    std::erase(tensor_nodes, this->ptr);

  return *this;
}

// --- Destructor
tf::tensor::~tensor() {
  if (tensor_nodes.end() !=
      std::find(tensor_nodes.begin(), tensor_nodes.end(), this->ptr)) {
    std::erase(tensor_nodes, this->ptr);
    if (this->ptr) {
      delete this->ptr;
      this->ptr = NULL;
    }
    if (opsPtr.size()) {
      for (void *opsptr : this->opsPtr)
        delete static_cast<Ops *>(opsptr);
      opsPtr.clear();
    }
  }
}

// --- Utility ---
void tf::tensor::assign_pointer(std::vector<unsigned> dimensions) {

  switch (this->dt_type) {
  case tf_float64:
    this->ptr = new Tensor<std::float64_t>(dimensions.size(), dimensions.data(),
                                           this->dt_type);
    tensor_nodes.push_back(this->ptr);
    break;
  default:
    ptr = nullptr;
  }
}

Tensor<std::float64_t> *tf::tensor::getPtr() { return this->ptr; }

unsigned tf::tensor::getNoOfDimensions() {
  return this->ptr->getNoOfDimensions();
}

const unsigned *tf::tensor::getDimensions() {
  return this->ptr->getDimensions();
}

unsigned tf::tensor::getNoOfElem() { return this->ptr->getNoOfElem(); }

void tf::tensor::tensor_of(double low_limit, double upper_limit) {

  switch (dt_type) {
  case tf_float64:
    this->ptr->initRandData(low_limit, upper_limit);
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
}

void tf::tensor::tensor_of(std::float64_t *data) {
  switch (dt_type) {
  case tf_float64:
    this->ptr->initData(data);
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
}

void tf::tensor::print_data() {
  switch (dt_type) {
  case tf_float64:
    this->ptr->printData();
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
}

void tf::tensor::print_dimension() {
  switch (dt_type) {
  case tf_float64:
    this->ptr->printDimensions();
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
}
// --- Utilty ---

// ---------------- Eager Mode ---------------
tf::tensor tf::tensor::matmul(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    opsPtr.push_back(new Opsmatmul);
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = this->ptr->matmul(
          *(input_b.getPtr()), std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, input_b.ptr);
    std::erase(tensor_nodes, output.ptr);
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
      output.ptr = this->ptr->add(*(input_b.getPtr()),
                                  std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, input_b.ptr);
    std::erase(tensor_nodes, output.ptr);
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
      output.ptr = this->ptr->add(*(input_b.getPtr()),
                                  std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, input_b.ptr);
    std::erase(tensor_nodes, output.ptr);
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
      output.ptr = this->ptr->mul(*(input_b.getPtr()),
                                  std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, input_b.ptr);
    std::erase(tensor_nodes, output.ptr);
  }
  return output;
}

tf::tensor tf::tensor::sigmoid() {
  tensor output;
  opsPtr.push_back(new Opssigmoid);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr =
        this->ptr->sigmoid(std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, output.ptr);
  }
  return output;
}

tf::tensor tf::tensor::scale(const std::float64_t scaleFactor) {
  tensor output;

  opsPtr.push_back(new Opsscale);
  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = this->ptr->scale(scaleFactor,
                                  std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, output.ptr);
  }
  return output;
}

tf::tensor tf::tensor::sqrt() {
  tensor output;

  opsPtr.push_back(new Opssqrt);
  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = this->ptr->sqrt(std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, output.ptr);
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
      output.ptr = this->ptr->sub(*(input_b.getPtr()),
                                  std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, input_b.ptr);
    std::erase(tensor_nodes, output.ptr);
  }
  return output;
}

tf::tensor tf::tensor::transpose() {
  tensor output;
  opsPtr.push_back(new Opstranspose);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr =
        this->ptr->transpose(std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, output.ptr);
  }
  return output;
}

tf::tensor tf::tensor::pow(const unsigned exponent) {
  tensor output;
  opsPtr.push_back(new Opspower);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr =
        this->ptr->pow(exponent, std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }

  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, output.ptr);
  }
  return output;
}

tf::tensor tf::tensor::relu() {
  tensor output;
  opsPtr.push_back(new Opsrelu);

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = this->ptr->relu(std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, output.ptr);
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
    output.ptr =
        this->ptr->mean(dim, std::span(opsPtr).subspan(opsPtr.size() - 2));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, output.ptr);
  }
  return output;
}

tf::tensor tf::tensor::mul(tensor &input_b) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    opsPtr.push_back(new Opsmul);
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output.ptr = this->ptr->mul(*(input_b.getPtr()),
                                  std::span(opsPtr).subspan(opsPtr.size() - 1));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, input_b.ptr);
    std::erase(tensor_nodes, output.ptr);
  }
  return output;
}

tf::tensor tf::tensor::getReduction(std::vector<unsigned> reduction_dims) {
  tensor output;
  opsPtr.push_back(new Opsreducesum);
  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output.ptr = this->ptr->reducesum(
        reduction_dims, std::span(opsPtr).subspan(opsPtr.size() - 1));
    break;

  default:
    break;
  }
  Graph *g = GraphManager::instance().getCurrentGraph();
  if (g) {
    std::erase(tensor_nodes, this->ptr);
    std::erase(tensor_nodes, output.ptr);
  }

  return output;
}
// ------------- End Eager Mode -------------

void tf::tensor::gradient_required(bool is_grad_required) {
  if (ptr) {
    this->ptr->gradientRequired(is_grad_required);
  }
}

// -------------- Graph Context ------------
tf::graph_context::graph_context() { this->graph_ctx = new GraphContext(); }

tf::graph_context::~graph_context() { delete graph_ctx; }

tf::tensor tf::graph_context::get_gradient(tensor &a) {
  Tensor<std::float64_t> *temp_ptr =
      static_cast<GraphContext *>(this->graph_ctx)
          ->graph_get_gradient(a.getPtr());

  tensor output(a.dt_type, temp_ptr);
  std::erase(tensor_nodes, output.ptr);

  return output;
}

void tf::graph_context::run() {
  static_cast<GraphContext *>(this->graph_ctx)->run();
}

void tf::graph_context::initialize_gradient() {
  static_cast<GraphContext *>(this->graph_ctx)->graph_initilize_gradient();
}

void tf::graph_context::compute_gradient() {
  static_cast<GraphContext *>(this->graph_ctx)->graph_compute_gradeint();
}
// -------------- Graph Context ------------

// --- Layers ---

// --- Dense ---
tf::layer::dense::dense(unsigned unit) { dense_layer = new Dense(unit); }

std::vector<tf::tensor>
tf::layer::dense::operator()(const std::vector<tf::tensor> &inputs) {
  std::vector<Tensor<std::float64_t> *> input_tensors;

  for (tf::tensor input : inputs) {
    input_tensors.push_back(input.ptr);
  }

  std::vector<Tensor<std::float64_t> *> output_tensors =
      (*this->dense_layer)(input_tensors);

  std::vector<tf::tensor> outputs;
  for (Tensor<std::float64_t> *output_tensor : output_tensors) {
    std::erase(tensor_nodes, output_tensor);
    outputs.push_back(tf::tensor(tf_float64, output_tensor));
  }
  return outputs;
}

std::vector<tf::tensor> tf::layer::dense::get_input_tensors() {
  std::vector<tf::tensor> input_tensors;

  std::vector<Tensor<std::float64_t> *> incoming_tensors =
      dense_layer->getInputTensors();

  for (Tensor<std::float64_t> *incoming_tensor : incoming_tensors)
    input_tensors.push_back(tf::tensor(tf_float64, incoming_tensor));
  return input_tensors;
}

std::vector<tf::tensor> tf::layer::dense::get_output_tensors() {
  std::vector<tf::tensor> output_tensors;

  std::vector<Tensor<std::float64_t> *> outgoing_tensors =
      dense_layer->getInputTensors();

  for (Tensor<std::float64_t> *outgoing_tensor : outgoing_tensors)
    output_tensors.push_back(tf::tensor(tf_float64, outgoing_tensor));
  return output_tensors;
}

void tf::layer::dense::set_weight(tf::tensor weight_tensor) {
  if (weight_tensor.ptr)
    static_cast<Dense *>(dense_layer)->setWeight(weight_tensor.ptr);
  else
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
}

void tf::layer::dense::set_bias(tf::tensor bias_tensor) {
  if (bias_tensor.ptr)
    static_cast<Dense *>(dense_layer)->setBias(bias_tensor.ptr);
  else
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
}

Layer *tf::layer::dense::getLayerPtr() { return this->dense_layer; }

// --- End of Dense ---

// --- End of Layers ---

// --- Model ---

tf::model::model(const std::vector<tf::tensor> &inputs,
                 const std::vector<tf::tensor> &outputs) {
  std::vector<Tensor<std::float64_t> *> input_tensors;
  std::vector<Tensor<std::float64_t> *> output_tensors;

  for (tf::tensor input : inputs)
    input_tensors.push_back(input.ptr);

  for (tf::tensor output : outputs)
    output_tensors.push_back(output.ptr);

  this->model_ptr = new Model(input_tensors, output_tensors);
}

void tf::model::fit(const std::vector<tf::tensor> &inputs,
                    const std::vector<tf::tensor> &outputs, callback call_back,
                    unsigned epochs, unsigned batch_size,
                    const std::vector<tf::tensor> &validation_datas,
                    unsigned verbose) {

  std::vector<Tensor<std::float64_t> *> input_tensors;
  std::vector<Tensor<std::float64_t> *> output_tensors;
  std::vector<Tensor<std::float64_t> *> validation_tensors;

  for (tf::tensor input : inputs) {
    std::erase(tensor_nodes, input.ptr);
    input_tensors.push_back(input.ptr);
  }

  for (tf::tensor output : outputs) {
    std::erase(tensor_nodes, output.ptr);
    output_tensors.push_back(output.getPtr());
  }

  for (tf::tensor validation_data : validation_datas) {
    std::erase(tensor_nodes, validation_data.ptr);
    validation_tensors.push_back(validation_data.getPtr());
  }

  this->model_ptr->fit(input_tensors, output_tensors, validation_tensors,
                       epochs, batch_size, call_back.getCallbackPtr(), verbose);
}

void tf::model::shuffle(bool shuffle) { model_ptr->shuffle(shuffle); }

// --- End Model ---

// --- Callback ---
tf::callback::callback(bool print_callback_log) {
  callback_ptr = new Callback(print_callback_log);
}

tf::callback::callback(unsigned callback_level) {
  callback_ptr = new Callback(callback_level);
}

void tf::callback::on_epoch_begin_get_trainable_parameter(
    tf::layer::dense dense_layer, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  callback_ptr->onEpochBeginGetTrainableParameter(
      dense_layer.getLayerPtr(), trainable_parameter_no, print_flag);
}

void tf::callback::on_epoch_end_get_trainable_parameter(
    tf::layer::dense dense_layer, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  callback_ptr->onEpochEndGetTrainableParameter(
      dense_layer.getLayerPtr(), trainable_parameter_no, print_flag);
}

std::vector<std::vector<tf::tensor>>
tf::callback::get_trainable_parameter_on_epoch_begin(
    tf::layer::dense dense_layer, Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<tf::tensor>> trainable_parametes_on_epoch_begin;

  std::vector<std::vector<Tensor<std::float64_t> *>> vector_vector_tensors =
      callback_ptr->getTrainableParameterEpochOnBegin(dense_layer.getLayerPtr(),
                                                      trainable_parameter_no);

  trainable_parametes_on_epoch_begin.resize(vector_vector_tensors.size());

  unsigned i = 0;
  for (unsigned i = 0; i < vector_vector_tensors.size(); i++) {
    for (Tensor<std::float64_t> *tensor : vector_vector_tensors[i])
      trainable_parametes_on_epoch_begin[i].push_back(
          tf::tensor(tf_float64, tensor));
  }

  return trainable_parametes_on_epoch_begin;
}

std::vector<std::vector<tf::tensor>>
tf::callback::get_trainable_parameter_on_epoch_end(
    tf::layer::dense dense_layer, Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<tf::tensor>> trainable_parametes_on_epoch_end;

  std::vector<std::vector<Tensor<std::float64_t> *>> vector_vector_tensors =
      callback_ptr->getTrainableParameterEpochOnEnd(dense_layer.getLayerPtr(),
                                                    trainable_parameter_no);

  trainable_parametes_on_epoch_end.resize(vector_vector_tensors.size());

  unsigned i = 0;
  for (unsigned i = 0; i < vector_vector_tensors.size(); i++) {
    for (Tensor<std::float64_t> *tensor : vector_vector_tensors[i]) {
      trainable_parametes_on_epoch_end[i].push_back(
          tf::tensor(tf_float64, tensor));
    }
  }

  return trainable_parametes_on_epoch_end;
}

Callback *tf::callback::getCallbackPtr() { return this->callback_ptr; }
// --- End Callback ---