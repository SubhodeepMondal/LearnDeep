#ifndef TENSOR_MAIN_API
#define TENSOR_MAIN_API

// C++ Headers
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// Library Headers
#include <core/framework/MathLibrary.h>
#include <core/graph/graph_context.hpp>
#include <core/ops/opskernel.h>

class Callback;
class Dense;
class Layer;
class Loss;
class Model;
class Optimizer;
struct LayerConfig;

namespace tf {
namespace optimizer {
class optimizer;
} // namespace optimizer

class tensor {
  // mutable std::vector<Ops *> opsPtr;
  Tensor<std::float64_t> *ptr{nullptr};

public:
  bool activateGraphSession{false};
  DataType dt_type;

  // --- Default constructor
  tensor();

  // --- Overloaded constructor
  tensor(DataType dt_type, const Tensor<std::float64_t> *ptr);

  // --- Destructor
  ~tensor();

  // --- Copy constructor
  tensor(const tensor &other);

  // --- Copy assignment
  tensor &operator=(const tensor &other);

  // --- Move constructor
  tensor(tensor &&other) noexcept;

  // --- Move assignment
  tensor &operator=(tensor &&other) noexcept;

  void addDimensions(std::vector<unsigned> &dimensions, unsigned w) {
    dimensions.push_back(w);
  }

  template <typename... args>
  void addDimensions(std::vector<unsigned> &dimensions, unsigned w,
                     args... Args) {
    addDimensions(dimensions, w);
    addDimensions(dimensions, Args...);
  }

  // --- Utility ---
  template <typename... Args> void tf_create(DataType d_type, Args... args) {
    unsigned *arr;
    std::vector<unsigned> dimensions;
    addDimensions(dimensions, args...);

    this->dt_type = d_type;
    assign_pointer(dimensions);
  }

  void tf_create(std::vector<unsigned> dims, DataType d_type);

  void assign_ptr(std::vector<unsigned> dimensions);

  void assign_ptr(void *ptr);

  tf::tensor copy() const;

  unsigned getNoOfDimensions();

  const unsigned *getDimensions() const;

  unsigned getNoOfElem();

  void tensor_of(double low_limit, double upper_limit);

  void tensor_of(std::float64_t *data);

  void print_data() const;

  void print_dimension();

  void assign_pointer(std::vector<unsigned> dimensions);

  void reshape(std::vector<unsigned> dimensions);
  // --- End Of Utility ---

  // ------- eager operations --------
  tensor operator+(tensor &input_b);

  tensor operator*(tensor &input_b);

  tensor add(tensor &input_b, bool graph_flag = true);

  tensor div(tensor &input_b, bool graph_flag = true);

  tensor greater_than_zero(bool graph_flag = true);

  tensor log(bool graph_flag = true);

  tensor mean(const unsigned dim, bool graph_flag = true);

  tensor matmul(const tensor &input_b, bool graph_flag = true) const;

  tensor mul(tensor &input_b, bool graph_flag = true);

  tensor pow(const unsigned exponent, bool graph_flag = true);

  tensor relu(bool graph_flag = true) const;

  tensor softmax(unsigned const axis = 0, bool graph_flag = true) const;

  tensor sigmoid(bool graph_flag = true);

  tensor scale(const std::float64_t scaleFactor, bool graph_flag = true);

  tensor sqrt(bool graph_flag = true);

  tensor sub(const tensor &input_b, bool graph_flag = true);

  tensor transpose(bool graph_flag = true);

  void gradient_required(bool is_grad_required);

  tensor reducesum(std::vector<unsigned> axis, bool keep_dims = false,
                   bool graph_flag = true) const;

  Tensor<std::float64_t> *getPtr() const;

  const void setPtr(Tensor<std::float64_t> *ptr);

  // Deletes the tensor when this wrapper owns it and drops its ownership;
  // non-owning aliases only drop the pointer. A spare-set tombstone remains so
  // graphs that still reference the address do not delete it a second time.
  void release() noexcept;

  std::float64_t *getData() const { return ptr->getData(); }
};

extern std::unordered_map<Tensor<std::float64_t> *, tensor *> tensor_nodes;
extern std::unordered_set<Tensor<std::float64_t> *> tensor_to_be_spared;

// Diagnostic count of tensors deleted by release(); used by lifecycle tests.
std::size_t released_tensor_count() noexcept;

using LayerParameterValue =
    std::variant<unsigned, double, std::string, tf::tensor>;

using LossParameterValue =
    std::variant<unsigned, double, std::string, tf::tensor,
                 std::vector<Tensor<std::float64_t> *>, std::vector<double>>;

using OptimizerParameterValue =
    std::variant<unsigned, double, std::string, tf::tensor>;

typedef struct graph_context {
private:
  GraphContext *graph_ctx;

public:
  graph_context();

  ~graph_context();

  void run();

  tensor get_gradient(tensor &a);

  void initialize_gradient();

  void inject_gradient(tf::tensor injection_for, tf::tensor injection_gradient);

  void compute_gradient();
} graph_context;

namespace layer {

using layer_id = std::uint64_t;
inline constexpr layer_id invalid_layer_id = 0;

namespace type {
inline constexpr std::string_view dense{"dense"};
inline constexpr std::string_view relu{"relu"};
inline constexpr std::string_view softmax{"softmax"};
} // namespace type

namespace parameter {
inline constexpr std::string_view input{"input"};
inline constexpr std::string_view weight{"weight"};
inline constexpr std::string_view bias{"bias"};
inline constexpr std::string_view matmul_result{"matmul_result"};
inline constexpr std::string_view output{"output"};
inline constexpr std::string_view training_input{"training_input"};
inline constexpr std::string_view training_weight{"training_weight"};
inline constexpr std::string_view training_bias{"training_bias"};
inline constexpr std::string_view training_matmul_result{
    "training_matmul_result"};
inline constexpr std::string_view training_output{"training_output"};
inline constexpr std::string_view grad_input{"grad_input"};
inline constexpr std::string_view grad_weight{"grad_weight"};
inline constexpr std::string_view grad_bias{"grad_bias"};
inline constexpr std::string_view incoming_grad{"incoming_grad"};
inline constexpr std::string_view updated_weight{"updated_weight"};
inline constexpr std::string_view updated_bias{"updated_bias"};
} // namespace parameter

class layer {
private:
  std::shared_ptr<Layer> layer_ptr;

  explicit layer(std::shared_ptr<Layer> layer_ptr);

protected:
  // Pointers into the owning layer's alias storage. The handle itself keeps no
  // input state, so these cannot dangle and cannot go stale during training.
  std::vector<const tf::tensor *> input_tensor_aliases() const;

public:
  layer();
  layer(std::string_view type_name);
  layer(std::string_view type_name, const LayerConfig &config);
  virtual ~layer();

  // Resolve a live layer from the public id space. Invalid ids and dead layers
  // produce an inert handle.
  static layer from_id(layer_id id);

  // Graph topology queries exposed through the public API so that consumers do
  // not need the internal LayerGraph ABI.
  static layer_id producer_of(const Tensor<std::float64_t> *tensor);
  static std::vector<layer_id>
  consumers_of(const Tensor<std::float64_t> *tensor);

  std::vector<tf::tensor> operator()(const std::vector<tf::tensor> &inputs);

  std::vector<tf::tensor> forward(const std::vector<tf::tensor> &inputs,
                                  unsigned batch_size);

  std::vector<tf::tensor> calculate(std::vector<tf::tensor> inputs);

  // Non-owning aliases whose getPtr() values identify the layer's real graph
  // tensors. Used by Model to build its execution graph.
  std::vector<tf::tensor> get_input_tensors() const;
  std::vector<tf::tensor> get_output_tensors() const;

  void initialize_parameters();

  void backward(const tf::optimizer::optimizer &optimizer);

  void reset_gradient_recording();

  std::unordered_set<std::string> parameterNames() const;

  std::vector<tf::tensor> getParameter(std::string_view name) const;

  void setParameter(std::string_view name, const LayerParameterValue &value);

  std::string_view type() const;

  // True when registry construction succeeded. Invalid handles are safe no-ops.
  explicit operator bool() const;

  bool valid() const;

  layer_id id() const;
};

struct dense : public layer {
  dense(unsigned unit);

  ~dense() override;

  std::vector<const tf::tensor *> get_input_tensors();

  std::vector<tf::tensor> get_output_tensors();

  std::vector<tf::tensor> get_output_training_tensor();

  std::vector<tf::tensor> get_output_training_weight();

  void set_weight(const tf::tensor &weight_tensor);

  void set_bias(const tf::tensor &bias_tensor);
};

struct relu : public layer {
  relu();

  ~relu() override;

  std::vector<const tf::tensor *> get_input_tensors();

  std::vector<tf::tensor> get_output_tensors();
};

struct softmax : public layer {
  softmax(unsigned axis = 0);

  ~softmax() override;

  std::vector<const tf::tensor *> get_input_tensors();

  std::vector<tf::tensor> get_output_tensors();
};

} // namespace layer

namespace loss {

using loss_id = std::uint64_t;
inline constexpr loss_id invalid_loss_id = 0;

namespace type {
inline constexpr std::string_view squared_error{"squared_error"};
inline constexpr std::string_view categorical_cross_entropy{
    "categorical_cross_entropy"};
} // namespace type

namespace parameter {
// Construction/config keys.
inline constexpr std::string_view inputs{"inputs"};
inline constexpr std::string_view name{"name"};

// Canonical parameter keys preserved across the string-based loss API.
inline constexpr std::string_view squared_error_predicted_output{
    "squared_error_predicted_output"};
inline constexpr std::string_view squared_error_target_output{
    "squared_error_target_output"};
inline constexpr std::string_view squared_error_grad_predicted_output{
    "squared_error_grad_predicted_output"};
inline constexpr std::string_view categorical_cross_entropy_error{
    "categorical_cross_entropy_error"};
inline constexpr std::string_view mean_categorical_cross_entropy_error{
    "mean_categorical_cross_entropy_error"};
inline constexpr std::string_view mean_absolute_categorical_cross_entropy_error{
    "mean_absolute_categorical_cross_entropy_error"};
inline constexpr std::string_view categorical_cross_entropy_input_grad{
    "categorical_cross_entropy_input_grad"};
inline constexpr std::string_view categorical_cross_entropy_target_output{
    "categorical_cross_entropy_target_output"};
} // namespace parameter

// Public loss configuration. This is the canonical type used by both the
// internal registry and out-of-tree loss plugins.
struct config {
  std::unordered_map<std::string, LossParameterValue> values;

  bool empty() const { return values.empty(); }

  bool has(const std::string &key) const { return values.count(key) != 0; }

  const LossParameterValue *get(const std::string &key) const {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
  }

  LossParameterValue *get(const std::string &key) {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
  }

  void set(std::string key, LossParameterValue value) {
    values.insert_or_assign(std::move(key), std::move(value));
  }

  template <typename T> const T *getAs(const std::string &key) const {
    const LossParameterValue *value = get(key);
    return value ? std::get_if<T>(value) : nullptr;
  }

  config &with(std::string key, LossParameterValue value) {
    set(std::move(key), std::move(value));
    return *this;
  }
};

// Public behavior contract for out-of-tree losses. A plugin implements this
// interface and calls registerBehavior(); the framework adapts it to the
// internal Loss ABI, so no framework-internal header is needed.
class behavior {
public:
  virtual ~behavior() = default;

  virtual std::string_view type() const = 0;
  virtual void forward(std::vector<tf::tensor *> inputs,
                       unsigned batch_size) = 0;
  virtual void backward() = 0;
  virtual void setTargetOutput(const std::vector<tf::tensor> &targets) = 0;
  virtual std::float64_t
  calculate(std::vector<tf::tensor> &predicted_output,
            std::vector<tf::tensor> &target_output) = 0;
  virtual std::float64_t getScalerLoss() = 0;
  virtual std::float64_t getEpochEndScalerLoss() { return getScalerLoss(); }
  virtual std::vector<tf::tensor *> getParameter(std::string_view name) = 0;
  virtual std::unordered_set<std::string> parameterNames() const = 0;
  virtual unsigned getNoOfInputs() const = 0;
};

using behavior_factory =
    std::function<std::unique_ptr<behavior>(const config &)>;
using behavior_setter = std::function<bool(
    behavior &, std::string_view, const LossParameterValue &)>;

bool registerBehavior(std::string_view name, behavior_factory factory,
                      behavior_setter setter = {});

// Public helper for exposing an input's autodiff gradient from a plugin without
// including the internal graph headers.
tf::tensor gradient_of(tf::tensor &input);

} // namespace loss

namespace optimizer {

class optimizer;

using optimizer_id = std::uint64_t;
inline constexpr optimizer_id invalid_optimizer_id = 0;

namespace type {
inline constexpr std::string_view sgd{"sgd"};
} // namespace type

namespace parameter {
inline constexpr std::string_view learning_rate{"learning_rate"};
} // namespace parameter

// Public optimizer configuration. This is the canonical type used by both the
// internal registry and future out-of-tree optimizer plugins.
struct config {
  std::unordered_map<std::string, OptimizerParameterValue> values;

  bool empty() const { return values.empty(); }

  bool has(const std::string &key) const { return values.count(key) != 0; }

  const OptimizerParameterValue *get(const std::string &key) const {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
  }

  OptimizerParameterValue *get(const std::string &key) {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
  }

  void set(std::string key, OptimizerParameterValue value) {
    values.insert_or_assign(std::move(key), std::move(value));
  }

  template <typename T> const T *getAs(const std::string &key) const {
    const OptimizerParameterValue *value = get(key);
    return value ? std::get_if<T>(value) : nullptr;
  }

  config &with(std::string key, OptimizerParameterValue value) {
    set(std::move(key), std::move(value));
    return *this;
  }
};

} // namespace optimizer
} // namespace tf

// Public self-registration macro. A plugin supplies a behavior_factory (and
// optionally a behavior_setter); the framework adapts it to the internal Loss
// ABI. The name REGISTER_LOSS is intentionally the same one used by in-tree
// losses, but its argument types belong entirely to the public API.
#define LD_LOSS_REGISTRATION_CONCAT_INNER(a, b) a##b
#define LD_LOSS_REGISTRATION_CONCAT(a, b)                                      \
  LD_LOSS_REGISTRATION_CONCAT_INNER(a, b)

#define REGISTER_LOSS(name, ...)                                               \
  namespace {                                                                  \
  [[maybe_unused]] const bool LD_LOSS_REGISTRATION_CONCAT(                     \
      ld_loss_registration_, __LINE__) = [] {                                  \
    const std::string loss_name{name};                                         \
    if (!tf::loss::registerBehavior(loss_name, __VA_ARGS__))                   \
      throw std::runtime_error("Failed to register loss: " + loss_name);       \
    return true;                                                               \
  }();                                                                         \
  } // namespace

using LossConfig = tf::loss::config;
using OptimizerConfig = tf::optimizer::config;

namespace tf {
namespace loss {

class loss {
private:
  std::shared_ptr<Loss> loss_ptr_;

public:
  loss() = default;

  loss(std::string_view type_name);

  loss(std::string_view type_name, const LossConfig &config);

  // Internal: wrap a registry-owned loss without creating a new instance.
  explicit loss(std::shared_ptr<Loss> owned);

  virtual ~loss() = default;

  void forward(std::vector<tf::tensor *> inputs, unsigned batch_size);

  void backward();

  std::float64_t calculate(std::vector<tf::tensor> &predicted_output,
                           std::vector<tf::tensor> &target_output);

  const std::float64_t get_loss();

  std::string get_loss_name();

  unsigned get_no_of_inputs();

  void set_target_output(std::vector<tf::tensor> target_output);

  std::unordered_set<std::string> parameterNames() const;

  std::vector<tf::tensor> getParameter(std::string_view name) const;

  void setParameter(std::string_view name, const LossParameterValue &value);

  std::string_view type() const;

  loss_id id() const;

  bool valid() const;

  explicit operator bool() const;
};

struct squared_error : public loss {
  squared_error(std::vector<Tensor<std::float64_t> *> input_preds,
                std::string loss_name = "mean_squared_loss");
};

struct categorical_cross_entropy : public loss {
  categorical_cross_entropy(
      std::vector<Tensor<std::float64_t> *> input_preds,
      std::string loss_name = "categorical_cross_entropy");
};

} // namespace loss

namespace callback {
typedef struct trace {
private:
  std::shared_ptr<Callback> callback_ptr;

public:
  trace();

  ~trace();

  void record_parameter_on_epoch_begin(
      const tf::layer::layer &layer_handle,
      const std::string &trainable_parameter, bool print_flag = false);

  void record_parameter_on_epoch_end(
      const tf::layer::layer &layer_handle,
      const std::string &trainable_parameter, bool print_flag = false);

  void record_parameter_on_batch_begin(
      const tf::layer::layer &layer_handle,
      const std::string &trainable_parameter, bool print_flag = false);

  void record_parameter_on_batch_end(
      const tf::layer::layer &layer_handle,
      const std::string &trainable_parameter, bool print_flag = false);

  void record_scalar_loss_on_epoch_end(tf::loss::loss loss);

  void record_tensor_loss_on_epoch_end(tf::loss::loss loss,
                                       std::string_view loss_parameter);

  void record_scalar_loss_on_batch_end(tf::loss::loss loss);

  void record_tensor_loss_on_batch_end(tf::loss::loss loss,
                                       std::string_view loss_parameter);

  std::vector<std::vector<tf::tensor>>
  get_parameter_on_epoch_begin(const tf::layer::layer &layer_handle,
                               const std::string &trainable_parameter);

  std::vector<std::vector<tf::tensor>>
  get_parameter_on_epoch_end(const tf::layer::layer &layer_handle,
                             const std::string &trainable_parameter);

  std::vector<std::vector<std::vector<tf::tensor>>>
  get_parameter_on_batch_begin(const tf::layer::layer &layer_handle,
                               const std::string &trainable_parameter);

  std::vector<std::vector<std::vector<tf::tensor>>>
  get_parameter_on_batch_end(const tf::layer::layer &layer_handle,
                             const std::string &trainable_parameter);

  std::vector<std::float64_t> get_scaler_loss_on_epoch_end(tf::loss::loss loss);

  std::vector<std::vector<tf::tensor>> get_tensor_loss_on_epoch_end(
      tf::loss::loss loss, std::string_view loss_parameter);

  std::vector<std::vector<std::float64_t>>
  get_scaler_loss_on_batch_end(tf::loss::loss loss);

  std::vector<std::vector<std::vector<tf::tensor>>>
  get_tensor_loss_on_batch_end(tf::loss::loss loss,
                               std::string_view loss_parameter);

  std::shared_ptr<Callback> callback() const;

} trace;

typedef struct earlystopping {
private:
  std::shared_ptr<Callback> callback_ptr;

public:
  earlystopping(tf::loss::loss loss, int patience, float min_delta,
                bool minimize);
  ~earlystopping() = default;

  std::shared_ptr<Callback> callback() const;
} earlystopping;

} // namespace callback
typedef struct history_container {
  std::unordered_map<std::string, std::vector<std::vector<std::float64_t>>>
      batch_history;

  /**
  type: loss, accuracy, recall, precision
  num_losses: huber loss, mse, squared_error, cse (there can be more than one
  epoch: number of epochs
  loss in  a model)
  value: loss value
   */
  std::unordered_map<
      std::string, std::unordered_map<std::string, std::vector<std::float64_t>>>
      history;

  void record_history_on_batch_end(unsigned const epoch_no,
                                   std::vector<tf::loss::loss_id> losses);

  void record_history_on_epoch_end(
      std::unordered_map<std::string, std::float64_t> epoch_loss);

  std::vector<std::float64_t> get_loss();

  std::vector<std::vector<std::float64_t>> operator[](std::string metric) {
    return batch_history[metric];
  }

} history_container;
typedef struct model {
private:
  Model *model_ptr;

  void
  model_fit(const std::vector<tf::tensor> &inputs,
            const std::vector<tf::tensor> &outputs,
            std::vector<Callback *> callback_ptr, unsigned epochs = 10,
            unsigned batch_size = 1,
            const std::vector<tf::tensor> &validation_datas = {tf::tensor()},
            unsigned verbose = 0);

public:
  model(const std::vector<tf::tensor> &inputs,
        const std::vector<tf::tensor> &outputs);

  ~model();

  /** @file model.hpp basic model implementation */
  /** @brief Updates hyperparamers for training */
  /** @param Optimizer class will be used for optimization */
  /** @param Loss loss to be used */
  /** @param Metric metric to be used */
  /** @return void */
  // void compile(tf::optimizer::optimizer = tf::optimizer::optimizer(),
  //              tf::loss::loss = tf::loss::loss(), tf::metric = tf::metric());

  /** @file model.hpp basic model implementation */
  /** @brief Updates hyperparamers for training */
  /** @param Optimizer class will be used for optimization */
  /** @param Loss loss to be used */
  /** @param Metric metric to be used */
  /** @return void */
  void compile(std::string_view optimizer_type, std::string_view loss_type);

  void compile(const tf::optimizer::optimizer &optimizer,
               std::string_view loss_type);

  /** @file tensor.cpp basic model implementation */
  /** @brief Runs the training loop and updates the gradients */
  /** @param inputs std vector inputs for the training */
  /** @param output expected output for the training, used for loss
   * calculation
   */
  /** @param epochs unsigned, default 10; required, no of iterations to run
   * for training and optimization */
  /** @param batch_size unsigned, default 1, batch size for each training */
  /** @param validation_data Tensor<std::float64_t> *,   output for the
   * training, used for loss calculation
   */
  /** @param callbacks unsigned default 0, learning rate schedular */
  /** @param verbose unsigned number training progress monitoring
   * level 0: default, none,
   * level 1: epoch progress
   * level 2: + loss
   * level 3: + metrics detail (i.e accuracy, precision, recall etc.)
   * level 4: + validation loss
   * level 5: + va;odation metric */
  /** @return void */
  history_container
  fit(const std::vector<tf::tensor> &inputs,
      const std::vector<tf::tensor> &outputs,
      std::vector<std::shared_ptr<Callback>> callbacks = {},
      unsigned epochs = 10, unsigned batch_size = 1,
      const std::vector<tf::tensor> &validation_datas = {tf::tensor()},
      unsigned verbose = 0, bool training = true);

  std::vector<tf::tensor> predict(const std::vector<tf::tensor> inputs);

  void shuffle(bool shuffle);

  tf::loss::loss get_model_loss(tf::tensor output) const;

  std::vector<layer::layer_id> get_layers();

  std::vector<tf::loss::loss_id> get_losses();

} model;
namespace optimizer {

// Shared-owning handle to an optimizer created through OptimizerRegistry.
// Invalid handles are safe no-ops for every operation.
class optimizer {
private:
  std::shared_ptr<Optimizer> optimizer_ptr_;

public:
  optimizer() = default;

  // String-key construction is the canonical entry point.
  optimizer(std::string_view type_name);
  optimizer(std::string_view type_name, const config &cfg);

  // Internal: wrap a registry-owned optimizer without creating a new instance.
  explicit optimizer(std::shared_ptr<Optimizer> owned);

  virtual ~optimizer() = default;

  void create_parameter_update_graph(tf::tensor param, tf::tensor &update_param,
                                     tf::tensor gradient_param) const;

  void execute() const;

  // Drops update ops from a previous training graph.
  void reset_update_graph() const;

  std::unordered_set<std::string> parameterNames() const;

  std::vector<tf::tensor> getParameter(std::string_view name) const;

  void setParameter(std::string_view name,
                    const OptimizerParameterValue &value);

  std::string_view type() const;

  optimizer_id id() const;

  bool valid() const;

  explicit operator bool() const;
};

} // namespace optimizer

typedef struct metric {

} metric;

} // namespace tf

#endif // TENSOR_MAIN_API
