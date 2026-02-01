#ifndef TENSOR_MAIN_API
#define TENSOR_MAIN_API

// C++ Headers
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <unordered_map>
#include <vector>

// Library Headers
#include <core/framework/MathLibrary.h>
#include <core/graph/graph_context.hpp>
#include <core/kernel/opskernel.h>
#include <layers/layer_enum.hpp>
#include <losses/loss_type.hpp>
#include <optimizers/optimizer_type.hpp>

class Callback;
class Dense;
class Layer;
class Loss;
class Model;
class Optimizer;

namespace tf {
typedef struct loss loss;
class tensor {
  // mutable std::vector<Ops *> opsPtr;
  Tensor<std::float64_t> *ptr{nullptr};

public:
  bool activateGraphSession;
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

  // tf::tensor deep_copy() const;

  void tf_create(std::vector<unsigned> dims, DataType d_type);

  void assign_ptr(std::vector<unsigned> dimensions);

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

  tensor mean(const unsigned dim, bool graph_flag = true);

  tensor matmul(const tensor &input_b, bool graph_flag = true) const;

  tensor mul(tensor &input_b, bool graph_flag = true);

  tensor pow(const unsigned exponent, bool graph_flag = true);

  tensor relu(bool graph_flag = true);

  tensor sigmoid(bool graph_flag = true);

  tensor scale(const std::float64_t scaleFactor, bool graph_flag = true);

  tensor sqrt(bool graph_flag = true);

  tensor sub(const tensor &input_b, bool graph_flag = true);

  tensor transpose(bool graph_flag = true);

  tensor getReduction(std::vector<unsigned> reduction_dims);

  void gradient_required(bool is_grad_required);

  template <typename... Args> tensor reducesum(Args... args) {
    std::vector<unsigned> dimensions;
    bool flag = true;

    // -------- end of eager operations ---------

    // Add dimensions to the vector
    addDimensions(dimensions, args...);

    unsigned *reduction_dims = new unsigned[dimensions.size()];
    for (int i = 0; i < dimensions.size(); i++) {
      reduction_dims[i] = dimensions[i];
    }

    delete[] reduction_dims;

    return getReduction(dimensions);
  }

  Tensor<std::float64_t> *getPtr() const;

  const void setPtr(Tensor<std::float64_t> *ptr);

  std::float64_t *getData() const { return ptr->getData(); }
};

extern std::unordered_map<Tensor<std::float64_t> *, tensor *> tensor_nodes;
extern std::unordered_set<Tensor<std::float64_t> *> tensor_to_be_spared;

typedef struct graph_context {
private:
  GraphContext *graph_ctx;

public:
  graph_context();

  ~graph_context();

  void run();

  tensor get_gradient(tensor &a);

  void initialize_gradient();

  void compute_gradient();
} graph_context;

namespace layer {

typedef struct dense {
private:
  Layer *dense_layer;

public:
  dense(unsigned unit);

  ~dense();

  std::vector<tf::tensor> operator()(const std::vector<tf::tensor> &inputs);

  std::vector<const tf::tensor *> get_input_tensors();

  std::vector<tf::tensor> get_output_tensors();

  std::vector<tf::tensor> get_output_training_tensor();

  std::vector<tf::tensor> get_output_training_weight();

  void set_weight(const tf::tensor &weight_tensor);

  void set_bias(const tf::tensor &bias_tensor);

  Layer *getLayerPtr() const;
} dense;

} // namespace layer

typedef struct callback {
private:
  Callback *callback_ptr;

public:
  callback(bool print_callback_log);

  callback(unsigned callback_level);

  ~callback();

  void record_parameter_on_epoch_begin(const layer::dense &dense_layer,
                                       Layer_Parameter trainable_parameter_no,
                                       bool print_flag = false);

  void record_parameter_on_epoch_end(const layer::dense &dense_layer,
                                     Layer_Parameter trainable_parameter_no,
                                     bool print_flag = false);

  void record_scalar_loss(tf::loss loss, bool printFlag = false);

  void record_tensor_loss(tf::loss loss, bool printFlag = false);

  void record_loss_tensoor(tf::loss loss, bool printFlag = false);

  std::vector<std::vector<tf::tensor>>
  get_parameter_on_epoch_begin(const layer::dense &dense_layer,
                               Layer_Parameter trainable_parameter_no);

  std::vector<std::vector<tf::tensor>>
  get_parameter_on_epoch_end(const layer::dense &dense_layer,
                             Layer_Parameter trainable_parameter_no);

  std::vector<std::float64_t> get_scaler_loss(tf::loss loss);

  std::vector<std::vector<tf::tensor>> get_tensor_loss(tf::loss loss);

  Callback *getCallbackPtr() const;

} callback;

typedef struct model {
private:
  Model *model_ptr;

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
  // void compile(tf::optimizer = tf::optimizer(), tf::loss = tf::loss(),
  //              tf::metric = tf::metric());

  /** @file model.hpp basic model implementation */
  /** @brief Updates hyperparamers for training */
  /** @param Optimizer class will be used for optimization */
  /** @param Loss loss to be used */
  /** @param Metric metric to be used */
  /** @return void */
  void compile(const OptimizerType optimizerType, const LossType lossType);

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
  void fit(const std::vector<tf::tensor> &inputs,
           const std::vector<tf::tensor> &outputs,
           const callback &call_back = callback(false), unsigned epochs = 10,
           unsigned batch_size = 1,
           const std::vector<tf::tensor> &validation_datas = {tf::tensor()},
           unsigned verbose = 0);

  void shuffle(bool shuffle);

  tf::loss get_model_loss(tf::tensor output) const;

} model;
typedef struct optimizer {
private:
  OptimizerType optimizer_type;
  Optimizer *optimizer_ptr;

public:
  optimizer() = default;

  optimizer(OptimizerType optimizer_type);

} optimizer;

typedef struct metric {

} metric;

typedef struct loss {

private:
  LossType loss_type;
  Loss *loss_ptr;

public:
  loss() = default;

  loss(const LossType lossType,
       std::vector<Tensor<std::float64_t> *> input_preds);

  void forward(std::vector<tf::tensor *> inputs, const unsigned batch_size);

  const std::float64_t get_loss();

  void set_target_output(std::vector<tf::tensor> target_output);

  Loss *const get_loss_ptr();

} loss;

} // namespace tf

#endif // TENSOR_MAIN_API