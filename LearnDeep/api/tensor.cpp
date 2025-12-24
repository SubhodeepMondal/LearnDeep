// Library Headers
#include "tensor.h"

// Third-party Headers
#include <absl/log/log.h>

// Library Headers
#include <callback/callback.hpp>
#include <layers/dense.hpp>
#include <model/model.hpp>

// --- Default Constructor
tf::tensor::tensor() : ptr(NULL) {}

tf::tensor::tensor(DataType dt_type, const Tensor<std::float64_t> *ptr) {
  if (ptr) {
    this->ptr = const_cast<Tensor<std::float64_t> *>(ptr);
    this->dt_type = dt_type;
  }
  tensor_nodes.insert(this->ptr);
}

// --- Copy constructor
tf::tensor::tensor(const tensor &other) {
  dt_type = other.dt_type;
  if (other.getPtr()) {
    this->ptr = other.getPtr();
  }
}

// --- Copy assignment
tf::tensor &tf::tensor::operator=(const tensor &other) {
  if (this != &other) {
    if (other.getPtr()) {
      if (this->ptr) {
        delete this->ptr;
        this->ptr = nullptr;
      }
      this->ptr = other.getPtr();
      this->dt_type = other.dt_type;
    }
  }
  return *this;
}

// --- Move constructor
tf::tensor::tensor(tensor &&other) noexcept {
  dt_type = other.dt_type;
  ptr = other.getPtr();
  other.ptr = nullptr;
}

// --- Move assignment
tf::tensor &tf::tensor::operator=(tensor &&other) noexcept {
  if (this != &other) {
    if (this->ptr)
      delete this->ptr;
    this->dt_type = other.dt_type;
    this->ptr = other.getPtr();
    other.ptr = nullptr;
  }

  return *this;
}

// --- Destructor
tf::tensor::~tensor() {
  if (tensor_nodes.count(this->ptr)) {
    tensor_nodes.erase(this->ptr);
    if (this->ptr) {
      delete this->ptr;
      this->ptr = NULL;
    }
  }
}

// --- Utility ---
tf::tensor tf::tensor::deep_copy() const {
  tf::tensor out;
  if (!this->ptr)
    return out;
  out.dt_type = this->dt_type;
  out.ptr = new Tensor<std::float64_t>(*this->ptr);

  tensor_nodes.insert(out.ptr);
  return out;
}

void tf::tensor::assign_pointer(std::vector<unsigned> dimensions) {

  switch (this->dt_type) {
  case tf_float64:
    this->ptr = new Tensor<std::float64_t>(dimensions.size(), dimensions.data(),
                                           this->dt_type);
    tensor_nodes.insert(this->ptr);
    break;
  default:
    ptr = nullptr;
  }
}

Tensor<std::float64_t> *tf::tensor::getPtr() const { return this->ptr; }

const void tf::tensor::setPtr(Tensor<std::float64_t> *ptr) { this->ptr = ptr; }

unsigned tf::tensor::getNoOfDimensions() {
  return this->ptr->getNoOfDimensions();
}

const unsigned *tf::tensor::getDimensions() const {
  return this->ptr->getDimensions();
}

unsigned tf::tensor::getNoOfElem() { return this->ptr->getNoOfElem(); }

void tf::tensor::tf_create(std::vector<unsigned> dimensions, DataType d_type) {
  this->dt_type = d_type;
  this->ptr = new Tensor<std::float64_t>(dimensions.size(), dimensions.data(),
                                         this->dt_type);
}

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

void tf::tensor::print_data() const {
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

void tf::tensor::reshape(std::vector<unsigned> dimensions) {
  if (this->ptr) {
    this->ptr->reshape(dimensions.size(), dimensions.data());
  }
}
// --- Utilty ---

// ---------------- Eager Mode ---------------
tf::tensor tf::tensor::matmul(const tensor &input_b) const {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output = tensor(this->dt_type, this->ptr->matmul(*(input_b.getPtr())));
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
      output = tensor(this->dt_type, this->ptr->add(*(input_b.getPtr())));
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
      output = tensor(this->dt_type, this->ptr->add(*(input_b.getPtr())));
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
      output = tensor(this->dt_type, this->ptr->mul(*(input_b.getPtr())));
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
    output = tensor(this->dt_type, this->ptr->sigmoid());
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
    output = tensor(this->dt_type, this->ptr->scale(scaleFactor));
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
    output = tensor(this->dt_type, this->ptr->sqrt());
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
      output = tensor(this->dt_type, this->ptr->sub(*(input_b.getPtr())));
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
    output = tensor(this->dt_type, this->ptr->transpose());
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
    output = tensor(this->dt_type, this->ptr->pow(exponent));
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
    output = tensor(this->dt_type, this->ptr->relu());
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
    output = tensor(this->dt_type, this->ptr->mean(dim));
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
      output = tensor(this->dt_type, this->ptr->mul(*(input_b.getPtr())));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::getReduction(std::vector<unsigned> reduction_dims) {
  tensor output;
  switch (dt_type) {
  case tf_float64:
    output = tensor(this->dt_type, this->ptr->reducesum(reduction_dims));
    break;

  default:
    break;
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

tf::graph_context::~graph_context() {
  graph_ctx->tensor_to_be_spared(tensor_nodes);
  delete graph_ctx;
}

tf::tensor tf::graph_context::get_gradient(tensor &a) {
  Tensor<std::float64_t> *temp_ptr =
      static_cast<GraphContext *>(this->graph_ctx)
          ->graph_get_gradient(a.getPtr());

  tensor output(a.dt_type, temp_ptr);
  tensor_nodes.erase(output.getPtr());

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
tf::layer::dense::dense(unsigned unit) {
  dense_layer = new Dense(unit);
  global_layer_graph.addNode(dense_layer);
}

tf::layer::dense::~dense() { delete this->dense_layer; }

std::vector<tf::tensor>
tf::layer::dense::operator()(const std::vector<tf::tensor> &inputs) {

  std::vector<tf::tensor> outputs = (*this->dense_layer)(inputs);

  return outputs;
}

std::vector<const tf::tensor *> tf::layer::dense::get_input_tensors() {
  return {nullptr};
}

std::vector<tf::tensor> tf::layer::dense::get_output_tensors() {
  return dense_layer->getOutputTensors();
}

void tf::layer::dense::set_weight(const tf::tensor &weight_tensor) {
  if (weight_tensor.getPtr()) {
    if (auto *d = dynamic_cast<Dense *>(dense_layer)) {
      d->setWeight(weight_tensor);
    } else {
      LOG(ERROR) << "Layer is not Dense";
    }
  } else
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
}

void tf::layer::dense::set_bias(const tf::tensor &bias_tensor) {
  if (bias_tensor.getPtr()) {
    if (auto *d = dynamic_cast<Dense *>(dense_layer)) {
      d->setBias(bias_tensor);
    } else {
      LOG(ERROR) << "Layer is not Dense";
    }
  } else
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
}

Layer *tf::layer::dense::getLayerPtr() const { return this->dense_layer; }

// --- End of Dense ---

// --- End of Layers ---

// --- Model ---

tf::model::model(const std::vector<tf::tensor> &inputs,
                 const std::vector<tf::tensor> &outputs) {
  this->model_ptr = new Model(inputs, outputs);
}

tf::model::~model() { delete model_ptr; }

void tf::model::fit(const std::vector<tf::tensor> &inputs,
                    const std::vector<tf::tensor> &outputs,
                    const callback &call_back, unsigned epochs,
                    unsigned batch_size,
                    const std::vector<tf::tensor> &validation_datas,
                    unsigned verbose) {

  this->model_ptr->fit(inputs, outputs, validation_datas, epochs, batch_size,
                       call_back.getCallbackPtr(), verbose);
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

tf::callback::~callback() { delete callback_ptr; }

void tf::callback::record_parameter_on_epoch_begin(
    const tf::layer::dense &dense_layer, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  callback_ptr->onEpochBeginGetTrainableParameter(
      dense_layer.getLayerPtr(), trainable_parameter_no, print_flag);
}

void tf::callback::record_parameter_on_epoch_end(
    const tf::layer::dense &dense_layer, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  callback_ptr->onEpochEndGetTrainableParameter(
      dense_layer.getLayerPtr(), trainable_parameter_no, print_flag);
}

std::vector<std::vector<tf::tensor>> tf::callback::get_parameter_on_epoch_begin(
    const tf::layer::dense &dense_layer,
    Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<tf::tensor>> trainable_parametes_on_epoch_begin;

  std::vector<std::vector<tf::tensor>> vector_vector_tensors =
      callback_ptr->getTrainableParameterEpochOnBegin(dense_layer.getLayerPtr(),
                                                      trainable_parameter_no);

  trainable_parametes_on_epoch_begin.resize(vector_vector_tensors.size());

  unsigned i = 0;
  for (unsigned i = 0; i < vector_vector_tensors.size(); i++) {
    for (tf::tensor tensor : vector_vector_tensors[i])
      trainable_parametes_on_epoch_begin[i].push_back(tensor);
  }

  return trainable_parametes_on_epoch_begin;
}

std::vector<std::vector<tf::tensor>> tf::callback::get_parameter_on_epoch_end(
    const tf::layer::dense &dense_layer,
    Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<tf::tensor>> trainable_parametes_on_epoch_end;

  std::vector<std::vector<tf::tensor>> vector_vector_tensors =
      callback_ptr->getTrainableParameterEpochOnEnd(dense_layer.getLayerPtr(),
                                                    trainable_parameter_no);

  trainable_parametes_on_epoch_end.resize(vector_vector_tensors.size());

  unsigned i = 0;
  for (unsigned i = 0; i < vector_vector_tensors.size(); i++) {
    for (tf::tensor tensor : vector_vector_tensors[i]) {
      trainable_parametes_on_epoch_end[i].push_back(tensor);
    }
  }

  return trainable_parametes_on_epoch_end;
}

Callback *tf::callback::getCallbackPtr() const { return this->callback_ptr; }
// --- End Callback ---
