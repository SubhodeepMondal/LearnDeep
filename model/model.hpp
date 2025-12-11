#ifndef _TENSOR_CORE_LAYER_
#define _TENSOR_CORE_LAYER_

// C++ Headers
#include <queue>
#include <stdfloat>
#include <unordered_map>
#include <vector>

// Library headers
#include <core/framework/MathLibrary.h>
#include <layers/layer_graph.hpp>

class Model {
private:
  std::vector<Tensor<std::float64_t> *> inputs;
  std::vector<Tensor<std::float64_t> *> outputs;

  std::unordered_map<Layer *, std::vector<Tensor<std::float64_t> *>>
      layer_input_mappings;
  std::unordered_map<Layer *, std::vector<Tensor<std::float64_t> *>>
      layer_output_mappings;

  std::unordered_map<Layer *, std::vector<Tensor<std::float64_t> *>>
      layer_training_input_mappings;

  std::unordered_map<Tensor<std::float64_t> *, std::vector<Layer *>>
      input_layer_mappings;

  LayerGraph model_layer_graph;
  std::vector<Layer *> layers;
  std::vector<Layer *> input_layers;

  std::vector<Tensor<std::float64_t> *>
  getLayers(std::queue<Tensor<std::float64_t> *> &output_queue);

  void getTrainingTensorsForInputLayer(
      const std::vector<Tensor<std::float64_t> *> &training_inputs);

  void doDummyAndTrainingTensorMapping(
      const std::vector<Tensor<std::float64_t> *> &training_inputs);

  void doTensorAndLayerMappings();

  void initializeLayerGraph();

public:
  Model(std::vector<Tensor<std::float64_t> *> const inputs);

  Model(const std::vector<Tensor<std::float64_t> *> &inputs,
        const std::vector<Tensor<std::float64_t> *> &outputs);

  /** @file model.hpp basic model implementation */
  /** @brief Updates hyperparamers for training */
  /** @param Optimizer class will be used for optimization */
  /** @param Loss loss to be used */
  /** @param Metric metric to be used */
  /** @return void */
  // void compile(tf::optimizer = tf::optimizer(), tf::loss = tf::loss(),
  //              tf::metric = tf::metric());

  /** @file model.hpp basic model implementation */
  /** @brief Runs the training loop and updates the gradients */
  /** @param inputs std vector inputs for the training */
  /** @param output expected output for the training, used for loss calculation
   */
  /** @param validation_data Tensor<std::float64_t> *,   output for the
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
  void fit(std::vector<Tensor<std::float64_t> *> inputs,
           std::vector<Tensor<std::float64_t> *> output,
           std::vector<Tensor<std::float64_t> *> valdiation_data = {NULL},
           unsigned epochs = 10, unsigned batch_size = 1, unsigned callback = 0,
           unsigned verbose = 0);

  /** @file model.hpp basic model implementation */
  /** @brief Feeds the model with input and does a inference on it to predict
   * output */
  /** @param input std vector of Tensor<std::float64_t> *, input tensor for the
   * model, of dimension no_feature x batch_size fearture can be of any
   * dimension */
  /** @return Tensor<std::float64_t> * output tensor after inference */
  Tensor<std::float64_t> *predict(std::vector<Tensor<std::float64_t> *> input);
};

#endif // _TENSOR_CORE_MODEL_