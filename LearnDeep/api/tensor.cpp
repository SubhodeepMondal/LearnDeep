// Library Headers
#include "tensor.h"

// Third-party Headers
#include <absl/log/log.h>

// Library Headers
#include <callback/callback.hpp>
#include <layers/layer_registry.hpp>
#include <losses/loss.hpp>
#include <model/model.hpp>
#include <numeric>
#include <optional>
#include <optimizers/optimizers.hpp>
#include <unordered_map>

class CallbackTrace;
class CallbackEarlyStopping;

std::unordered_map<Tensor<std::float64_t> *, tf::tensor *> tf::tensor_nodes;
std::unordered_set<Tensor<std::float64_t> *> tf::tensor_to_be_spared;

namespace {

void store_input_aliases(std::vector<tf::tensor> &aliases,
                         const std::vector<tf::tensor> &inputs) {
  aliases.clear();
  aliases.reserve(inputs.size());

  for (const tf::tensor &input : inputs)
    aliases.push_back(input);
}

std::vector<const tf::tensor *>
input_alias_ptrs(const std::vector<tf::tensor> &aliases) {
  std::vector<const tf::tensor *> inputs;
  inputs.reserve(aliases.size());

  for (const tf::tensor &alias : aliases)
    inputs.push_back(&alias);

  return inputs;
}

std::vector<tf::tensor>
copy_output_tensors(const std::vector<tf::tensor *> &tensors) {
  std::vector<tf::tensor> copies;
  copies.reserve(tensors.size());

  for (const tf::tensor *tensor : tensors)
    if (tensor)
      copies.push_back(*tensor);

  return copies;
}

LayerConfig config_with(const std::string &key, unsigned value) {
  LayerConfig config;
  config.set(key, value);
  return config;
}

const std::unordered_map<std::string,
                         std::unordered_map<std::string, Layer_Parameter>> &
layer_parameter_maps() {
  static const std::unordered_map<
      std::string, std::unordered_map<std::string, Layer_Parameter>>
      maps = {
          {"dense",
           {{"input", Layer_Parameter::dense_input},
            {"weight", Layer_Parameter::dense_weight},
            {"bias", Layer_Parameter::dense_bias},
            {"matmul_result", Layer_Parameter::dense_matmul_result},
            {"output", Layer_Parameter::dense_output},
            {"training_input", Layer_Parameter::dense_training_input},
            {"training_weight", Layer_Parameter::dense_training_weight},
            {"training_bias", Layer_Parameter::dense_training_bias},
            {"training_matmul_result",
             Layer_Parameter::dense_training_matmul_result},
            {"training_output", Layer_Parameter::dense_training_output},
            {"grad_input", Layer_Parameter::dense_grad_input},
            {"grad_weight", Layer_Parameter::dense_grad_weight},
            {"grad_bias", Layer_Parameter::dense_grad_bias},
            {"incoming_grad", Layer_Parameter::dense_incoming_grad},
            {"updated_weight", Layer_Parameter::dense_updated_weight},
            {"updated_bias", Layer_Parameter::dense_updated_bias}}},
          {"relu",
           {{"grad_input", Layer_Parameter::relu_grad_input},
            {"input", Layer_Parameter::relu_input},
            {"output", Layer_Parameter::relu_output},
            {"training_input", Layer_Parameter::relu_training_input},
            {"training_output", Layer_Parameter::relu_training_output}}},
          {"softmax",
           {{"grad_input", Layer_Parameter::softmax_grad_input},
            {"input", Layer_Parameter::softmax_input},
            {"output", Layer_Parameter::softmax_output},
            {"training_input", Layer_Parameter::softmax_training_input},
            {"training_output", Layer_Parameter::softmax_training_output}}}};

  return maps;
}

std::optional<Layer_Parameter> layer_parameter_for(const std::string &type,
                                                   const std::string &name) {
  const auto &maps = layer_parameter_maps();
  auto type_it = maps.find(type);
  if (type_it == maps.end())
    return std::nullopt;

  auto parameter_it = type_it->second.find(name);
  if (parameter_it == type_it->second.end())
    return std::nullopt;

  return parameter_it->second;
}

} // namespace

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
  this->activateGraphSession = other.activateGraphSession;
  dt_type = other.dt_type;
  if (other.getPtr()) {
    this->ptr = other.getPtr();
  }
}

// --- Copy assignment
tf::tensor &tf::tensor::operator=(const tensor &other) {
  if (this != &other) {
    this->activateGraphSession = other.activateGraphSession;
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
  this->activateGraphSession = other.activateGraphSession;
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
    this->activateGraphSession = other.activateGraphSession;
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

void tf::tensor::assign_ptr(void *ptr) {

  switch (this->dt_type) {
  case tf_float64:
    this->ptr = static_cast<Tensor<std::float64_t> *>(ptr);
    break;
  default:
    this->ptr = nullptr;
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

tf::tensor tf::tensor::copy() const {
  tf::tensor new_tensor;
  new_tensor.dt_type = this->dt_type;
  new_tensor.assign_ptr(new Tensor<std::float64_t>(*this->ptr));
  return new_tensor;
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

tf::tensor tf::tensor::div(tensor &input_b, bool graph_flag) {
  tensor output;

  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      output = tensor(this->dt_type,
                      this->ptr->div(*(input_b.getPtr()), graph_flag));
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

tf::tensor tf::tensor::log(bool graph_flag) {
  tensor output;

  switch (dt_type) {
  case tf_float64: {
    output = tensor(this->dt_type, this->ptr->log(graph_flag));
    break;
  }
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

tf::tensor tf::tensor::softmax(unsigned const axis, bool graph_flag) const {
  tensor output;

  switch (dt_type) {
  case tf_float64:
    output.dt_type = this->dt_type;
    output = tensor(this->dt_type, this->ptr->softmax(axis, graph_flag));
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

tf::tensor tf::tensor::reducesum(std::vector<unsigned> axis, bool keep_dims,
                                 bool graph_flag) const {
  tensor output;
  switch (dt_type) {
  case tf_float64:
    output = tensor(this->dt_type,
                    this->ptr->reducesum(axis, keep_dims, graph_flag));
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

void tf::graph_context::inject_gradient(tf::tensor injection_for,
                                        tf::tensor injection_gradient) {
  static_cast<GraphContext *>(this->graph_ctx)
      ->inject_gradient(injection_for.getPtr(), injection_gradient.getPtr());
}

void tf::graph_context::compute_gradient() {
  static_cast<GraphContext *>(this->graph_ctx)->graph_compute_gradeint();
}
// -------------- Graph Context ------------

// --- Layers ---

// --- Generic layer handle ---

tf::layer::layer::layer() = default;

tf::layer::layer::layer(const std::string &type_name)
    : layer(type_name, LayerConfig{}) {}

tf::layer::layer::layer(const std::string &type_name,
                        const LayerConfig &config)
    : type_name_(type_name) {
  std::unique_ptr<Layer> created =
      LayerRegistry::instance().create(type_name, config);
  if (!created) {
    this->type_name_.clear();
    return;
  }

  Layer *raw_layer = created.release();
  global_layer_graph.addNode(raw_layer);
  this->layer_ptr = std::shared_ptr<Layer>(raw_layer, [](Layer *layer) {
    if (!layer)
      return;
    global_layer_graph.removeNode(layer);
    delete layer;
  });
}

tf::layer::layer::~layer() = default;

std::vector<tf::tensor>
tf::layer::layer::operator()(const std::vector<tf::tensor> &inputs) {
  store_input_aliases(this->input_aliases, inputs);

  if (!this->layer_ptr)
    return {};

  return copy_output_tensors((*this->layer_ptr)(inputs));
}

std::vector<tf::tensor>
tf::layer::layer::forward(const std::vector<tf::tensor> &inputs,
                          unsigned batch_size) {
  store_input_aliases(this->input_aliases, inputs);

  if (!this->layer_ptr)
    return {};

  std::vector<const tf::tensor *> raw_inputs;
  raw_inputs.reserve(this->input_aliases.size());
  for (const tf::tensor &input : this->input_aliases)
    raw_inputs.push_back(&input);

  return copy_output_tensors(this->layer_ptr->forward(raw_inputs, batch_size));
}

std::vector<tf::tensor>
tf::layer::layer::calculate(std::vector<tf::tensor> inputs) {
  store_input_aliases(this->input_aliases, inputs);

  if (!this->layer_ptr)
    return {};

  return this->layer_ptr->calculate(inputs);
}

void tf::layer::layer::initialize_parameters() {
  if (this->layer_ptr)
    this->layer_ptr->initializeParameters();
}

std::unordered_set<std::string> tf::layer::layer::parameterNames() const {
  const auto &maps = layer_parameter_maps();
  auto type_it = maps.find(this->type_name_);
  if (type_it == maps.end())
    return {};

  std::unordered_set<std::string> names;
  for (const auto &parameter : type_it->second)
    names.insert(parameter.first);
  return names;
}

std::vector<tf::tensor>
tf::layer::layer::getParameter(const std::string &name) const {
  if (!this->layer_ptr)
    return {};

  auto parameter = layer_parameter_for(this->type_name_, name);
  if (!parameter) {
    LOG(ERROR) << "Fatal! Parameter '" << name
               << "' is not available for layer type '" << this->type_name_
               << "'.\n";
    return {};
  }

  return copy_output_tensors(
      this->layer_ptr->getLayerParameter(*parameter));
}

void tf::layer::layer::setParameter(const std::string &name,
                                    const LayerParameterValue &value) {
  if (!this->layer_ptr)
    return;

  LayerRegistry::instance().setParameter(this->type_name_,
                                         this->layer_ptr.get(), name, value);
}

std::string tf::layer::layer::type() const { return this->type_name_; }

Layer *tf::layer::layer::getLayerPtr() const { return this->layer_ptr.get(); }

const void *tf::layer::layer::identity() const {
  return static_cast<const void *>(this->layer_ptr.get());
}

// --- End of Generic layer handle ---

// --- Dense ---

tf::layer::dense::dense(unsigned unit)
    : layer("dense", config_with("units", unit)) {}

tf::layer::dense::~dense() = default;

std::vector<const tf::tensor *> tf::layer::dense::get_input_tensors() {
  return input_alias_ptrs(this->input_aliases);
}

std::vector<tf::tensor> tf::layer::dense::get_output_tensors() {
  return this->getParameter("output");
}

std::vector<tf::tensor> tf::layer::dense::get_output_training_tensor() {
  return this->getParameter("training_output");
}

std::vector<tf::tensor> tf::layer::dense::get_output_training_weight() {
  return this->getParameter("training_weight");
}

void tf::layer::dense::set_weight(const tf::tensor &weight_tensor) {
  if (!weight_tensor.getPtr()) {
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
    return;
  }

  this->setParameter("weight", weight_tensor);
}

void tf::layer::dense::set_bias(const tf::tensor &bias_tensor) {
  if (!bias_tensor.getPtr()) {
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
    return;
  }

  this->setParameter("bias", bias_tensor);
}

// --- End of Dense ---

// --- Relu ---

tf::layer::relu::relu() : layer("relu") {}

tf::layer::relu::~relu() = default;

std::vector<const tf::tensor *> tf::layer::relu::get_input_tensors() {
  return input_alias_ptrs(this->input_aliases);
}

std::vector<tf::tensor> tf::layer::relu::get_output_tensors() {
  return this->getParameter("output");
}

// --- End of Relu ---

// --- Softmax ---

tf::layer::softmax::softmax(unsigned axis)
    : layer("softmax", config_with("axis", axis)) {}

tf::layer::softmax::~softmax() = default;

std::vector<const tf::tensor *> tf::layer::softmax::get_input_tensors() {
  return input_alias_ptrs(this->input_aliases);
}

std::vector<tf::tensor> tf::layer::softmax::get_output_tensors() {
  return this->getParameter("output");
}

// --- End of Softmax ---
// --- End of Layers ---

// --- Model ---

tf::model::model(const std::vector<tf::tensor> &inputs,
                 const std::vector<tf::tensor> &outputs) {
  this->model_ptr = new Model(inputs, outputs);
}

tf::model::~model() { delete model_ptr; }

tf::history_container
tf::model::fit(const std::vector<tf::tensor> &inputs,
               const std::vector<tf::tensor> &outputs,
               std::vector<std::shared_ptr<Callback>> callback_ptr,
               unsigned epochs, unsigned batch_size,
               const std::vector<tf::tensor> &validation_datas,
               unsigned verbose, bool training) {

  return this->model_ptr->fit(*this, inputs, outputs, validation_datas, epochs,
                              batch_size, callback_ptr, verbose, training);
}

std::vector<tf::tensor>
tf::model::predict(const std::vector<tf::tensor> inputs) {

  return this->model_ptr->predict(inputs);
}

void tf::model::shuffle(bool shuffle) { model_ptr->shuffle(shuffle); }

void tf::model::compile(const OptimizerType optimizerType,
                        const LossType lossType) {
  this->model_ptr->compile(optimizerType, lossType);
}

tf::loss tf::model::get_model_loss(tf::tensor output) const {
  return this->model_ptr->getModelLoss(output.getPtr());
}

std::vector<Layer *> &tf::model::get_layers() { return model_ptr->getLayers(); }

std::vector<Loss *> tf::model::get_losses() {
  std::vector<Loss *> losses;
  for (tf::loss *loss : model_ptr->getLosses()) {
    losses.push_back(loss->get_loss_ptr());
  }
  return losses;
}
// --- End Model ---

// --- Callback ---
tf::callback::trace::trace() {
  this->callback_ptr = std::make_shared<CallbackTrace>(false);
}

tf::callback::trace::~trace() {}

void tf::callback::trace::record_parameter_on_epoch_begin(
    Layer *const layer_ptr, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->onEpochBeginGetTrainableParameter(layer_ptr, trainable_parameter_no,
                                          print_flag);
}

void tf::callback::trace::record_parameter_on_epoch_end(
    Layer *const layer_ptr, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->onEpochEndGetTrainableParameter(layer_ptr, trainable_parameter_no,
                                        print_flag);
}

void tf::callback::trace::record_parameter_on_batch_begin(
    Layer *const layer_ptr, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordTrainableParameterOnBatchBegin(layer_ptr, trainable_parameter_no,
                                             print_flag);
}

void tf::callback::trace::record_parameter_on_batch_end(
    Layer *const layer_ptr, Layer_Parameter trainable_parameter_no,
    bool print_flag) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordTrainableParameterOnBatchEnd(layer_ptr, trainable_parameter_no,
                                           print_flag);
}

void tf::callback::trace::record_scalar_loss_on_epoch_end(tf::loss loss) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordScalerLossEpochEnd(loss.get_loss_ptr());
}

void tf::callback::trace::record_tensor_loss_on_epoch_end(
    tf::loss loss, Loss_Parameter loss_parameter) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordTensorLossEpochEnd(loss.get_loss_ptr(), loss_parameter);
}

void tf::callback::trace::record_scalar_loss_on_batch_end(tf::loss loss) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordScalerLossBatchEnd(loss.get_loss_ptr());
}

void tf::callback::trace::record_tensor_loss_on_batch_end(
    tf::loss loss, Loss_Parameter loss_parameter) {
  static_cast<CallbackTrace *>(this->callback_ptr.get())
      ->recordTensorLossBatchEnd(loss.get_loss_ptr(), loss_parameter);
}

std::vector<std::vector<tf::tensor>>
tf::callback::trace::get_parameter_on_epoch_begin(
    Layer *const layer_ptr, Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<tf::tensor>> trainable_parametes_on_epoch_begin;

  std::vector<std::vector<tf::tensor *>> vector_vector_tensors =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getTrainableParameterEpochOnBegin(layer_ptr,
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
    Layer *const layer_ptr, Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<tf::tensor>> trainable_parametes_on_epoch_end;

  std::vector<std::vector<tf::tensor *>> vector_vector_tensors =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getTrainableParameterEpochOnEnd(layer_ptr, trainable_parameter_no);

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
    Layer *const layer_ptr, Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<std::vector<tf::tensor>>>
      trainable_parametes_on_batch_begin;

  std::vector<std::vector<std::vector<tf::tensor *>>> vector_vector_tensors =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getTrainableParameterBatchOnBegin(layer_ptr,
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
    Layer *const layer_ptr, Layer_Parameter trainable_parameter_no) {
  std::vector<std::vector<std::vector<tf::tensor>>>
      trainable_parametes_on_batch_end;

  std::vector<std::vector<std::vector<tf::tensor *>>> vector_vector_tensors =
      static_cast<CallbackTrace *>(this->callback_ptr.get())
          ->getTrainableParameterBatchOnEnd(layer_ptr, trainable_parameter_no);

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
    std::unordered_map<std::string, std::float64_t> epoch_loss) {

  for (const auto &[key, value] : epoch_loss) {
    history["loss"][key].push_back(value);
  }
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
               std::vector<Tensor<std::float64_t> *> input_preds,
               std::string loss_name) {
  switch (lossType) {
  case (LossType::squared_error): {
    this->loss_ptr = new SquaredError(input_preds, loss_name);
    break;
  }
  case (LossType::categorical_cross_entropy): {
    this->loss_ptr = new CategoricalCrossEntropy(input_preds, loss_name);
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

std::float64_t tf::loss::calculate(std::vector<tf::tensor> &predicted_ouput,
                                   std::vector<tf::tensor> &target_output) {
  return this->loss_ptr->calculate(predicted_ouput, target_output);
}

const std::float64_t tf::loss::get_loss() {
  return this->loss_ptr->getScalerLoss();
}

std::string tf::loss::get_loss_name() { return this->loss_ptr->getLossName(); }

unsigned tf::loss::get_no_of_inputs() {
  return this->loss_ptr->getNoOfInputs();
}

void tf::loss::set_target_output(std::vector<tf::tensor> target_output) {
  this->loss_ptr->setTargetOutput(target_output);
}

Loss *const tf::loss::get_loss_ptr() { return this->loss_ptr; }
// --- End Losses --
