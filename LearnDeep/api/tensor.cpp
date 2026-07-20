// Library Headers
#include "tensor.h"

// Third-party Headers
#include <absl/log/log.h>

// Library Headers
#include <callback/callback.hpp>
#include <layers/dense.hpp>
#include <layers/relu.hpp>
#include <losses/loss.hpp>
#include <model/model.hpp>
#include <numeric>
#include <optimizers/optimizers.hpp>

class CallbackTrace;
class CallbackEarlyStopping;

std::unordered_map<Tensor<std::float64_t> *, tf::tensor *> tf::tensor_nodes;
std::unordered_set<Tensor<std::float64_t> *> tf::tensor_to_be_spared;

// --- Default Constructor
tf::tensor::tensor() : ptr(NULL) {}

tf::tensor::tensor(DataType dt_type, const Tensor<std::float64_t> *ptr) {
  if (ptr) {
    if (this->ptr)
      if (tf::tensor_nodes[this->ptr] == this) {
        tf::tensor_nodes.erase(this->ptr);
        if (this->ptr) {
          delete this->ptr;
          this->ptr = NULL;
        }
      }
    this->ptr = const_cast<Tensor<std::float64_t> *>(ptr);
    this->dt_type = dt_type;
    tf::tensor_nodes[this->ptr] = this;
    tf::tensor_to_be_spared.insert(this->ptr);
  }
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
  this->dt_type = other.dt_type;
  this->ptr = other.getPtr();
  tf::tensor_nodes[this->ptr] = this; // steal ownership of the Tensor
  other.ptr = nullptr;
}

// --- Move assignment
tf::tensor &tf::tensor::operator=(tensor &&other) noexcept {
  if (this != &other) {
    if (this->ptr)
      delete this->ptr;
    this->dt_type = other.dt_type;
    this->ptr = other.getPtr();
    tf::tensor_nodes[this->ptr] = this; // steal ownership of the Tensor
    other.ptr = nullptr;
  }

  return *this;
}

// --- Destructor
tf::tensor::~tensor() {
  if (this->ptr) {
    if (tf::tensor_nodes[this->ptr] == this) {
      delete this->ptr;
      this->ptr = NULL;
    }
  }
}

void tf::tensor::assign_pointer(std::vector<unsigned> dimensions) {

  switch (this->dt_type) {
  case tf_float64:
    if (this->ptr) {
      if (tf::tensor_nodes[this->ptr] == this) {
        tf::tensor_nodes.erase(this->ptr);
        delete this->ptr;
        this->ptr = NULL;
      }
    }
    this->ptr = new Tensor<std::float64_t>(dimensions.size(), dimensions.data(),
                                           this->dt_type);
    tf::tensor_nodes[this->ptr] = this;
    tf::tensor_to_be_spared.insert(this->ptr);
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

  if (this->ptr) {
    if (tf::tensor_nodes[this->ptr] == this) {
      tf::tensor_nodes.erase(this->ptr);
      delete this->ptr;
      this->ptr = NULL;
    }
  }
  this->dt_type = d_type;
  this->ptr = new Tensor<std::float64_t>(dimensions.size(), dimensions.data(),
                                         this->dt_type);
  tf::tensor_nodes[this->ptr] = this;
  tf::tensor_to_be_spared.insert(this->ptr);
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
tf::tensor tf::tensor::matmul(const tensor &input_b, bool graph_flag) const {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output = tensor(this->dt_type,
                      this->ptr->matmul(*(input_b.getPtr()), graph_flag));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::add(tensor &input_b, bool graph_flag) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output = tensor(this->dt_type,
                      this->ptr->add(*(input_b.getPtr()), graph_flag));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::greater_than_zero(bool graph_flag) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output = tensor(this->dt_type, this->ptr->greaterThanZero(graph_flag));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
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

tf::tensor tf::tensor::sigmoid(bool graph_flag) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output = tensor(this->dt_type, this->ptr->sigmoid(graph_flag));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  return output;
}

tf::tensor tf::tensor::scale(const std::float64_t scaleFactor,
                             bool graph_flag) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output = tensor(this->dt_type, this->ptr->scale(scaleFactor, graph_flag));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  return output;
}

tf::tensor tf::tensor::sqrt(bool graph_flag) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output = tensor(this->dt_type, this->ptr->sqrt(graph_flag));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  return output;
}

tf::tensor tf::tensor::sub(const tensor &input_b, bool graph_flag) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output = tensor(this->dt_type,
                      this->ptr->sub(*(input_b.getPtr()), graph_flag));
      break;
    }
    default:
      LOG(ERROR) << "Invalid data type!";
    }
  }
  return output;
}

tf::tensor tf::tensor::transpose(bool graph_flag) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output = tensor(this->dt_type, this->ptr->transpose(graph_flag));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  return output;
}

tf::tensor tf::tensor::pow(const unsigned exponent, bool graph_flag) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output = tensor(this->dt_type, this->ptr->pow(exponent, graph_flag));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  return output;
}

tf::tensor tf::tensor::relu(bool graph_flag) const {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output = tensor(this->dt_type, this->ptr->relu(graph_flag));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  return output;
}

tf::tensor tf::tensor::mean(const unsigned dim, bool graph_flag) {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output = tensor(this->dt_type, this->ptr->mean(dim, graph_flag));
    break;
  default:
    LOG(ERROR) << "Invalid data type!";
  }
  return output;
}

tf::tensor tf::tensor::mul(tensor &input_b, bool graph_flag) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output.dt_type = this->dt_type;
      output = tensor(this->dt_type,
                      this->ptr->mul(*(input_b.getPtr()), graph_flag));
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
  // std::unordered_set<Tensor<std::float64_t> *> tensor_nodes_to_be_spared;
  // for (auto tensor : tf::tensor_nodes) {
  //   tensor_nodes_to_be_spared.insert(tensor.first);
  //   std::cout << tensor.first << ", ";
  // }
  // std::cout << std::endl;
  graph_ctx->tensor_to_be_spared(tf::tensor_to_be_spared);
  delete graph_ctx;
}

tf::tensor tf::graph_context::get_gradient(tensor &a) {
  Tensor<std::float64_t> *temp_ptr =
      static_cast<GraphContext *>(this->graph_ctx)
          ->graph_get_gradient(a.getPtr());

  tensor output(a.dt_type, temp_ptr);
  // tensor_nodes.erase(output.getPtr());

  return output;
}

void tf::graph_context::run() {
  static_cast<GraphContext *>(this->graph_ctx)->run();
}

void tf::graph_context::initialize_gradient() {
  static_cast<GraphContext *>(this->graph_ctx)->graph_initialize_gradient();
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

tf::layer::dense::~dense() { delete dynamic_cast<Dense *>(this->dense_layer); }

std::vector<tf::tensor>
tf::layer::dense::operator()(const std::vector<tf::tensor> &inputs) {
  std::vector<tf::tensor> layer_outputs;

  const std::vector<tf::tensor *> &outputs = (*this->dense_layer)(inputs);

  for (const tf::tensor *output : outputs)
    layer_outputs.push_back(*output);

  return layer_outputs;
}

std::vector<const tf::tensor *> tf::layer::dense::get_input_tensors() {
  return {nullptr};
}

std::vector<tf::tensor> tf::layer::dense::get_output_tensors() {
  std::vector<tf::tensor> output_tensors;

  std::vector<tf::tensor *> temp_output_tensors =
      dense_layer->getOutputTensors();

  for (tf::tensor *tensor : temp_output_tensors)
    output_tensors.push_back(*tensor);

  return output_tensors;
}

void tf::layer::dense::set_weight(const tf::tensor &weight_tensor) {
  if (weight_tensor.getPtr()) {
    static_cast<Dense *>(this->dense_layer)->setWeight(weight_tensor);
  } else
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
}

void tf::layer::dense::set_bias(const tf::tensor &bias_tensor) {
  if (bias_tensor.getPtr()) {
    static_cast<Dense *>(this->dense_layer)->setBias(bias_tensor);
  } else
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
}

Layer *tf::layer::dense::getLayerPtr() const { return this->dense_layer; }

// --- End of Dense ---

// --- Relu ---

tf::layer::relu::relu() {
  this->relu_layer = new Relu();
  global_layer_graph.addNode(this->relu_layer);
}

tf::layer::relu::~relu() { delete dynamic_cast<Relu *>(this->relu_layer); }

std::vector<tf::tensor>
tf::layer::relu::operator()(const std::vector<tf::tensor> &inputs) {

  std::vector<tf::tensor> layer_outputs;

  const std::vector<tf::tensor *> &outputs = (*this->relu_layer)(inputs);

  for (const tf::tensor *output : outputs)
    layer_outputs.push_back(*output);

  return layer_outputs;
}

std::vector<const tf::tensor *> tf::layer::relu::get_input_tensors() {
  return {nullptr};
}

std::vector<tf::tensor> tf::layer::relu::get_output_tensors() {
  std::vector<tf::tensor> output_tensors;

  std::vector<tf::tensor *> temp_output_tensors =
      relu_layer->getOutputTensors();

  for (tf::tensor *tensor : temp_output_tensors)
    output_tensors.push_back(*tensor);

  return output_tensors;
}
// --- End of Relu ---

// --- End of Layers ---

// --- Model ---

tf::model::model(const std::vector<tf::tensor> &inputs,
                 const std::vector<tf::tensor> &outputs) {
  this->model_ptr = new Model(inputs, outputs);
}

tf::model::~model() { delete model_ptr; }

std::unordered_map<std::string, std::vector<std::float64_t>>
tf::model::fit(const std::vector<tf::tensor> &inputs,
               const std::vector<tf::tensor> &outputs,
               std::vector<std::shared_ptr<Callback>> callback_ptr,
               unsigned epochs, unsigned batch_size,
               const std::vector<tf::tensor> &validation_datas,
               unsigned verbose) {

  return this->model_ptr->fit(inputs, outputs, validation_datas, epochs,
                              batch_size, callback_ptr, verbose);
}

void tf::model::shuffle(bool shuffle) { model_ptr->shuffle(shuffle); }

void tf::model::compile(const OptimizerType optimizerType,
                        const LossType lossType) {
  this->model_ptr->compile(optimizerType, lossType);
}

tf::loss tf::model::get_model_loss(tf::tensor output) const {
  return this->model_ptr->getModelLoss(output.getPtr());
}
// --- End Model ---

// --- Callback ---
tf::callback::trace::trace() {
  this->callback_ptr = std::make_shared<CallbackTrace>(false);
}

tf::callback::trace::~trace() {}

void tf::callback::trace::record_parameter_on_epoch_begin(
    const tf::layer::dense &dense_layer, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->onEpochBeginGetTrainableParameter(dense_layer.getLayerPtr(),
                                          trainable_parameter_no, print_flag);
}

void tf::callback::trace::record_parameter_on_epoch_end(
    const tf::layer::dense &dense_layer, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->onEpochEndGetTrainableParameter(dense_layer.getLayerPtr(),
                                        trainable_parameter_no, print_flag);
}

void tf::callback::trace::record_parameter_on_batch_begin(
    const tf::layer::dense &dense_layer, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordTrainableParameterOnBatchBegin(
          dense_layer.getLayerPtr(), trainable_parameter_no, print_flag);
}

void tf::callback::trace::record_parameter_on_batch_end(
    const tf::layer::dense &dense_layer, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordTrainableParameterOnBatchEnd(dense_layer.getLayerPtr(),
                                           trainable_parameter_no, print_flag);
}

void tf::callback::trace::record_scalar_loss_on_epoch_end(tf::loss loss,
                                                          bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordScalerLossEpochEnd(loss.get_loss_ptr(), print_flag);
}

void tf::callback::trace::record_tensor_loss_on_epoch_end(
    tf::loss loss, Loss_Parameter loss_parameter, bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordTensorLossEpochEnd(loss.get_loss_ptr(), loss_parameter,
                                 print_flag);
}

void tf::callback::trace::record_scalar_loss_on_batch_end(tf::loss loss,
                                                          bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordScalerLossBatchEnd(loss.get_loss_ptr(), print_flag);
}

void tf::callback::trace::record_tensor_loss_on_batch_end(
    tf::loss loss, Loss_Parameter loss_parameter, bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordTensorLossBatchEnd(loss.get_loss_ptr(), loss_parameter,
                                 print_flag);
}

std::vector<std::vector<tf::tensor>>
tf::callback::trace::get_parameter_on_epoch_begin(
    const tf::layer::dense &dense_layer,
    Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<tf::tensor>> trainable_parametes_on_epoch_begin;

  std::vector<std::vector<tf::tensor *>> vector_vector_tensors =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getTrainableParameterEpochOnBegin(dense_layer.getLayerPtr(),
                                              trainable_parameter_no);

  trainable_parametes_on_epoch_begin.resize(vector_vector_tensors.size());

  unsigned i = 0;
  for (unsigned i = 0; i < vector_vector_tensors.size(); i++) {
    for (tf::tensor *tensor : vector_vector_tensors[i])
      trainable_parametes_on_epoch_begin[i].push_back(*tensor);
  }

  return trainable_parametes_on_epoch_begin;
}

std::vector<std::vector<tf::tensor>>
tf::callback::trace::get_parameter_on_epoch_end(
    const tf::layer::dense &dense_layer,
    Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<tf::tensor>> trainable_parametes_on_epoch_end;

  std::vector<std::vector<tf::tensor *>> vector_vector_tensors =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getTrainableParameterEpochOnEnd(dense_layer.getLayerPtr(),
                                            trainable_parameter_no);

  trainable_parametes_on_epoch_end.resize(vector_vector_tensors.size());

  unsigned i = 0;
  for (unsigned i = 0; i < vector_vector_tensors.size(); i++) {
    for (tf::tensor *tensor : vector_vector_tensors[i]) {
      trainable_parametes_on_epoch_end[i].push_back(*tensor);
    }
  }

  return trainable_parametes_on_epoch_end;
}

std::vector<std::vector<std::vector<tf::tensor>>>
tf::callback::trace::get_parameter_on_batch_begin(
    const tf::layer::dense &dense_layer,
    Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<std::vector<tf::tensor>>>
      trainable_parametes_on_batch_begin;

  std::vector<std::vector<std::vector<tf::tensor *>>> vector_vector_tensors =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getTrainableParameterBatchOnBegin(dense_layer.getLayerPtr(),
                                              trainable_parameter_no);

  trainable_parametes_on_batch_begin.resize(vector_vector_tensors.size());

  for (unsigned j = 0; j < vector_vector_tensors.size(); j++) {
    trainable_parametes_on_batch_begin[j].resize(
        vector_vector_tensors[j].size());
    for (unsigned i = 0; i < vector_vector_tensors[j].size(); i++) {
      for (tf::tensor *tensor : vector_vector_tensors[j][i])
        trainable_parametes_on_batch_begin[j][i].push_back(*tensor);
    }
  }

  return trainable_parametes_on_batch_begin;
}

std::vector<std::vector<std::vector<tf::tensor>>>
tf::callback::trace::get_parameter_on_batch_end(
    const tf::layer::dense &dense_layer,
    Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<std::vector<tf::tensor>>>
      trainable_parametes_on_batch_end;

  std::vector<std::vector<std::vector<tf::tensor *>>> vector_vector_tensors =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getTrainableParameterBatchOnEnd(dense_layer.getLayerPtr(),
                                            trainable_parameter_no);

  trainable_parametes_on_batch_end.resize(vector_vector_tensors.size());

  for (unsigned j = 0; j < vector_vector_tensors.size(); j++) {
    trainable_parametes_on_batch_end[j].resize(vector_vector_tensors[j].size());
    for (unsigned i = 0; i < vector_vector_tensors[j].size(); i++) {
      for (tf::tensor *tensor : vector_vector_tensors[j][i])
        trainable_parametes_on_batch_end[j][i].push_back(*tensor);
    }
  }

  return trainable_parametes_on_batch_end;
}

std::vector<std::float64_t>
tf::callback::trace::get_scaler_loss_on_epoch_end(tf::loss loss) {
  return static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->getScalerLossEpochEnd(loss.get_loss_ptr());
}

std::vector<std::vector<tf::tensor>>
tf::callback::trace::get_tensor_loss_on_epoch_end(
    tf::loss loss, Loss_Parameter loss_parameter) {
  std::vector<std::vector<tf::tensor>> tensor_losses;

  std::vector<std::vector<tf::tensor *>> tensor_loss_ptrs =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getLossParameterEpochEnd(loss.get_loss_ptr(), loss_parameter);

  tensor_losses.resize(tensor_loss_ptrs.size());
  unsigned i = 0;
  for (std::vector<tf::tensor *> epochs : tensor_loss_ptrs) {
    for (tf::tensor *tensor_ptr : epochs)
      tensor_losses[i++].push_back(*tensor_ptr);
  }
  return tensor_losses;
}

std::vector<std::vector<std::float64_t>>
tf::callback::trace::get_scaler_loss_on_batch_end(tf::loss loss) {
  return static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->getScalerLossBatchEnd(loss.get_loss_ptr());
}

std::vector<std::vector<std::vector<tf::tensor>>>
tf::callback::trace::get_tensor_loss_on_batch_end(
    tf::loss loss, Loss_Parameter loss_parameter) {
  std::vector<std::vector<std::vector<tf::tensor>>> tensor_losses;

  std::vector<std::vector<std::vector<tf::tensor *>>> tensor_loss_ptrs =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getLossParameterBatchEnd(loss.get_loss_ptr(), loss_parameter);

  tensor_losses.resize(tensor_loss_ptrs.size());
  for (unsigned i = 0; i < tensor_loss_ptrs.size(); i++) {
    tensor_losses[i].resize(tensor_loss_ptrs[i].size());
    for (unsigned j = 0; j < tensor_loss_ptrs[i].size(); j++)
      for (tf::tensor *tensor_ptr : tensor_loss_ptrs[i][j])
        tensor_losses[i][j].push_back(*tensor_ptr);
  }
  return tensor_losses;
}

std::shared_ptr<Callback> tf::callback::trace::callback() const {
  return this->callback_ptr;
}

tf::callback::earlystopping::earlystopping(tf::loss loss, int patience,
                                           float min_delta, bool minimize) {
  this->callback_ptr = std::make_shared<CallbackEarlyStopping>(
      loss.get_loss_ptr(), patience, min_delta, minimize);
}

std::shared_ptr<Callback> tf::callback::earlystopping::callback() const {
  return callback_ptr;
}
// --- End Callback ...

// --- History ---

void tf::history_container::record_history_on_batch_end(
    unsigned const epoch_no, const std::vector<tf::loss *> losses) {
  std::float64_t batch_total_loss(0.0);

  if (this->batch_history["loss"].size() == epoch_no)
    this->batch_history["loss"].resize(epoch_no + 1);

  for (auto loss_ptr : losses)
    batch_total_loss += loss_ptr->get_loss();

  this->batch_history["loss"][epoch_no].push_back(batch_total_loss);
}

void tf::history_container::record_history_on_epoch_end(
    unsigned const no_of_batches) {
  unsigned index = batch_history["loss"].size() - 1;

  std::float64_t sum =
      std::accumulate(this->batch_history["loss"][index].begin(),
                      this->batch_history["loss"][index].end(), 0.0);

  history["loss"].push_back(sum / no_of_batches);
}

// --- End History ---

// --- Optimizer ---
tf::optimizer::optimizer(OptimizerType optimizerType) {
  switch (optimizerType) {
  case (OptimizerType::SGD): {
    this->optimizer_ptr = new SGD();
    break;
  }
  }
}

Optimizer *tf::optimizer::getPtr() { return optimizer_ptr; }

void tf::optimizer::execute_optimizer() {
  this->optimizer_ptr->executeOptimizer();
}
// --- End Optimizer ---

// --- Losses ---
tf::loss::loss(const LossType lossType,
               std::vector<Tensor<std::float64_t> *> input_preds) {
  switch (lossType) {
  case (LossType::squared_error): {
    this->loss_ptr = new SquaredError(input_preds);
    break;
  }
  }
}

void tf::loss::forward(std::vector<tf::tensor *> inputs,
                       const unsigned batch_size) {
  if (this->loss_ptr) {
    loss_ptr->forward(inputs, batch_size);
  }
}

void tf::loss::backward() {
  if (this->loss_ptr) {
    loss_ptr->backward();
  }
}

const std::float64_t tf::loss::get_loss() {
  return this->loss_ptr->getScalerLoss();
}

void tf::loss::set_target_output(std::vector<tf::tensor> target_output) {
  this->loss_ptr->setTargetOutput(target_output);
}

Loss *const tf::loss::get_loss_ptr() { return this->loss_ptr; }
// --- End Losses --