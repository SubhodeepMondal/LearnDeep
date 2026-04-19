#ifndef _TENSOR_CORE_LAYER_
#define _TENSOR_CORE_LAYER_

// C++ Headers
#include <stdfloat>
#include <unordered_map>
#include <vector>

// Library headers
#include <api/tensor.h>
#include <layers/layer_graph.hpp>

class Model {
private:
  unsigned batch_size;
  bool shuffle_input;
  bool auto_grad_created;
  std::vector<Tensor<std::float64_t> *> inputs;
  std::vector<Tensor<std::float64_t> *> outputs;

  tf::tensor *temp_training_input_tensor;

  std::vector<tf::tensor *> local_training_inputs;

  std::unordered_map<Layer *, std::vector<const Tensor<std::float64_t> *>>
      layer_input_mappings;
  std::unordered_map<Layer *, std::vector<const tf::tensor *>>
      layer_output_mappings;

  std::unordered_map<Layer *, std::vector<const tf::tensor *>>
      layer_training_input_mappings;
  std::unordered_map<const Tensor<std::float64_t> *, std::vector<Layer *>>
      input_layer_mappings;

  LayerGraph model_layer_graph;
  std::vector<Layer *> layers;
  std::vector<Layer *> input_layers;

  tf::optimizer optimizer;
  std::vector<tf::loss *> losses;
  std::unordered_map<tf::loss *,
                     std::vector<Tensor<std::float64_t> *>>
      loss_input_mappings; // this maps loss and all the input tensor(aka target
                           // output) that is feed
  std::unordered_map<tf::loss *, std::vector<tf::tensor *>>
      loss_training_input_mappings;
  //   tf::metric metric;

  std::vector<const Tensor<std::float64_t> *> getLayersNBackTrackInputs(
      std::queue<const Tensor<std::float64_t> *> &output_queue);

  void initializeTrainingMappings();

  std::vector<tf::tensor>
  createBatchTargetBuffer(const std::vector<tf::tensor> &training_target);

  unsigned selectBatchIndex(int &randIndex, unsigned batch_count);

  void loadTrainingBatch(const std::vector<tf::tensor> &training_inputs,
                         unsigned batch_index);

  void runCallbackOnEpochBegin(std::vector<std::shared_ptr<Callback>> callback);

  void runCallbackOnEpochEnd(std::vector<std::shared_ptr<Callback>> callback);

  void runCallbackOnBatchBegin(std::vector<std::shared_ptr<Callback>> callback,
                               unsigned const epoch_no);

  void runCallbackOnBatchEnd(std::vector<std::shared_ptr<Callback>> callback,
                             unsigned const epoch_no);

  bool checkEarlyStopping(std::vector<std::shared_ptr<Callback>> callbacks);

  void setTrainingTensorsForInputLayer();

  void setTargetOutputForLoss(const std::vector<tf::tensor> &training_target,
                              std::vector<tf::tensor> &local_temp_outputs,
                              unsigned index);

  void doDummyAndTrainingTensorMapping();

  void doTensorAndLayerMappings();

  void initializeLayerGraph();

  void
  initilizeInputsForTraining(const std::vector<tf::tensor> &training_inputs);

public:
  Model(std::vector<tf::tensor> const inputs);

  Model(const std::vector<tf::tensor> &inputs,
        const std::vector<tf::tensor> &outputs);

  ~Model();

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
  void compile(const OptimizerType optimizer_type, const LossType loss_type);

  /** @file model.hpp basic model implementation */
  /** @brief Runs the training loop and updates the gradients */
  /** @param inputs std vector inputs for the training */
  /** @param output expected output for the training, used for loss calculation
   */
  /** @param validation_data tf::tensor ,   output for the
   * training, used for loss calculation
   */
  /** @param epochs unsigned, default 10; required, no of iterations to run for
   * training and optimization */
  /** @param batch_size unsigned, default 1, batch size for each training */
  /** @param callbacks unsigned default 0, learning rate schedular */
  /** @param verbose unsigned number training progress monitoring
   * level 0: default, none,
   * level 1: epoch progress
   * level 2: + loss
   * level 3: + metrics detail (i.e accuracy, precision, recall etc.)
   * level 4: + validation loss
   * level 5: + va;odation metric */
  /** @return void */
  std::unordered_map<std::string, std::vector<std::float64_t>>
  fit(const std::vector<tf::tensor> &inputs,
      const std::vector<tf::tensor> &output,
      const std::vector<tf::tensor> &valdiation_data = {}, unsigned epochs = 10,
      unsigned batch_size = 1,
      std::vector<std::shared_ptr<Callback>> callback =
          std::vector<std::shared_ptr<Callback>>(),
      unsigned verbose = 0);

  /** @file model.hpp basic model implementation */
  /** @brief Feeds the model with input and does a inference on it to predict
   * output */
  /** @param input std vector of tf::tensor , input tensor for the
   * model, of dimension no_feature x batch_size fearture can be of any
   * dimension */
  /** @return tf::tensor  output tensor after inference */
  tf::tensor predict(std::vector<tf::tensor> input);

  void shuffle(bool shuffle);

  tf::loss getModelLoss(Tensor<std::float64_t> *output_tensor);
};

#endif // _TENSOR_CORE_MODEL_
