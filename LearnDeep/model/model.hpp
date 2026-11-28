#ifndef _TENSOR_CORE_LAYER_
#define _TENSOR_CORE_LAYER_

// C++ Headers
#include <memory>
#include <queue>
#include <stdfloat>
#include <string_view>
#include <unordered_map>
#include <vector>

// Library headers
#include <api/tensor.h>

class Loss;

class Model {
private:
  // Construction builds the layer graph in a temporary state and commits it
  // only after validation succeeds. Keeping these fields together prevents a
  // failed model from leaving partially populated mapping tables behind.
  struct GraphConstructionState {
    std::unordered_map<tf::layer::layer_id,
                       std::vector<const Tensor<std::float64_t> *>>
        layer_input_mappings;
    std::unordered_map<tf::layer::layer_id, std::vector<tf::tensor>>
        layer_output_mappings;
    std::unordered_map<const Tensor<std::float64_t> *,
                       std::vector<tf::layer::layer_id>>
        input_layer_mappings;
    std::unordered_map<tf::layer::layer_id, tf::layer::layer> owned_layers;
    std::vector<tf::layer::layer_id> layers;
    std::vector<tf::layer::layer_id> input_layers;
    std::vector<tf::layer::layer_id> execution_order;
  };

  unsigned batch_size;
  bool shuffle_input;
  bool auto_grad_created;
  bool pred_mapping;
  std::vector<Tensor<std::float64_t> *> inputs;
  std::vector<Tensor<std::float64_t> *> outputs;
  std::vector<tf::tensor> prediction_outputs;

  // Owned batch buffers; local_training_inputs holds non-owning views into
  // these tensors for the mapping tables.
  std::vector<std::unique_ptr<tf::tensor>> owned_training_inputs;
  std::vector<tf::tensor *> local_training_inputs;

  std::unordered_map<tf::layer::layer_id,
                     std::vector<const Tensor<std::float64_t> *>>
      layer_input_mappings;
  std::unordered_map<tf::layer::layer_id, std::vector<tf::tensor>>
      layer_output_mappings;

  std::unordered_map<tf::layer::layer_id, std::vector<const tf::tensor *>>
      layer_training_input_mappings;
  // Stable aliases returned by tf::layer::layer::forward(); downstream mappings
  // point into these vectors instead of into the layer's internal ABI.
  std::unordered_map<tf::layer::layer_id, std::vector<tf::tensor>>
      layer_training_output_mappings;
  std::unordered_map<tf::layer::layer_id, std::vector<tf::tensor>>
      layer_prediction_input_mappings;

  std::unordered_map<const Tensor<std::float64_t> *,
                     std::vector<tf::layer::layer_id>>
      input_layer_mappings;

  // The model owns public layer handles; it never reaches through to the
  // internal Layer ABI.
  std::unordered_map<tf::layer::layer_id, tf::layer::layer> owned_layers;
  std::vector<tf::layer::layer_id> layers; // all the layers for the model
  std::vector<tf::layer::layer_id>
      input_layers; // layers that are receiving inputs
  // Validated topological order, computed once from the layer graph. Execution
  // follows this order instead of retrying layers whose inputs are not ready.
  std::vector<tf::layer::layer_id> execution_order;

  tf::layer::layer lockLayer(tf::layer::layer_id id) const;

  static std::vector<tf::layer::layer_id>
  buildExecutionOrder(const GraphConstructionState &graph,
                      const std::vector<Tensor<std::float64_t> *> &inputs);

  void commitGraphConstructionState(GraphConstructionState &graph);

  tf::optimizer::optimizer optimizer;

  // Losses are owned through the registry and referenced everywhere else by
  // their stable id. This removes dangling raw pointers from callbacks while
  // still allowing foreign-loss rejection and stale-id pruning.
  std::vector<tf::loss::loss_id> losses;
  std::unordered_map<tf::loss::loss_id, std::shared_ptr<Loss>> owned_losses;
  std::unordered_map<tf::loss::loss_id,
                     std::vector<Tensor<std::float64_t> *>>
      loss_input_mappings; // this maps loss and all the input tensor(aka target
                           // output) that is feed
  std::unordered_map<tf::loss::loss_id, std::vector<tf::tensor *>>
      loss_training_input_mappings;

  std::shared_ptr<Loss> lockLoss(tf::loss::loss_id id) const;
  //   tf::metric metric;

  std::vector<const Tensor<std::float64_t> *> getLayersNBackTrackInputs(
      std::queue<const Tensor<std::float64_t> *> &output_queue,
      GraphConstructionState &graph);

  void initializeTrainingMappings();

  // Clears state that is only valid for one training graph.
  void resetTrainingGraphState();

  void initializePredictionInputMappings();

  std::vector<tf::tensor>
  createBatchTargetBuffer(const std::vector<tf::tensor> &training_target);

  std::unordered_map<std::string, std::float64_t>
  getEpochlevelLosses(std::vector<tf::tensor> &predicted_output,
                      const std::vector<tf::tensor> &target_output);

  unsigned selectBatchIndex(int &randIndex, unsigned batch_count);

  void loadTrainingBatch(const std::vector<tf::tensor> &training_inputs,
                         unsigned batch_index);

  bool runCallbackBeforeTraining(
      const std::vector<std::shared_ptr<Callback>> &callbacks,
      tf::model &model);

  void runCallbackAfterTraining(
      const std::vector<std::shared_ptr<Callback>> &callbacks);

  void runCallbackOnEpochBegin(
      const std::vector<std::shared_ptr<Callback>> &callbacks);

  void runCallbackOnEpochEnd(
      const std::vector<std::shared_ptr<Callback>> &callbacks);

  void runCallbackOnBatchBegin(
      const std::vector<std::shared_ptr<Callback>> &callbacks,
      unsigned const epoch_no);

  void
  runCallbackOnBatchEnd(const std::vector<std::shared_ptr<Callback>> &callbacks,
                        unsigned const epoch_no);

  bool
  checkEarlyStopping(const std::vector<std::shared_ptr<Callback>> &callbacks);

  void setTrainingTensorsForInputLayer();

  void setTargetOutputForLoss(const std::vector<tf::tensor> &training_target,
                              std::vector<tf::tensor> &local_temp_outputs,
                              unsigned index);

  void doDummyAndTrainingTensorMapping();

  void doLayerCalculation();

  void doTensorAndLayerMappings(GraphConstructionState &graph);

  void
  initilizeInputsForTraining(const std::vector<tf::tensor> &training_inputs);

  void initializeInputsForPrediction(const std::vector<tf::tensor> &inputs);

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
  // void compile(tf::optimizer::optimizer = tf::optimizer::optimizer(),
  //              tf::loss::loss = tf::loss::loss(), tf::metric = tf::metric());

  /** @file model.hpp basic model implementation */
  /** @brief Updates hyperparamers for training */
  /** @param Optimizer string key of the optimizer to construct */
  /** @param Loss loss to be used */
  /** @return void */
  void compile(std::string_view optimizer_type, std::string_view loss_type);

  /** @brief Compile with a preconfigured optimizer handle. */
  void compile(const tf::optimizer::optimizer &optimizer,
               std::string_view loss_type);

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
  tf::history_container fit(tf::model &model,
                            const std::vector<tf::tensor> &inputs,
                            const std::vector<tf::tensor> &output,
                            const std::vector<tf::tensor> &valdiation_data = {},
                            unsigned epochs = 10, unsigned batch_size = 1,
                            std::vector<std::shared_ptr<Callback>> callback =
                                std::vector<std::shared_ptr<Callback>>(),
                            unsigned verbose = 0, bool training = true);

  /** @file model.hpp basic model implementation */
  /** @brief Feeds the model with input and does a inference on it to predict
   * output */
  /** @param input std vector of tf::tensor , input tensor for the
   * model, of dimension no_feature x batch_size fearture can be of any
   * dimension */
  /** @return tf::tensor  output tensor after inference */
  std::vector<tf::tensor> predict(std::vector<tf::tensor> inputs);

  void shuffle(bool shuffle);

  tf::loss::loss getModelLoss(Tensor<std::float64_t> *output_tensor);

  std::vector<tf::loss::loss_id> getLosses() const { return this->losses; }

  std::vector<tf::layer::layer_id> getLayers() const { return this->layers; }
};

#endif // _TENSOR_CORE_MODEL_
