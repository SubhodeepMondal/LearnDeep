// C++ Headers
#include <algorithm>
#include <iterator>
#include <queue>
#include <stdexcept>

// Library Headers
#include "model.hpp"
#include <absl/log/log.h>
#include <callback/callback.hpp>
#include <core/utility/initializers.hpp>
#include <layers/layer_graph.hpp>
#include <layers/layers.hpp>
#include <losses/loss.hpp>
#include <losses/loss_registry.hpp>
#include <optimizers/optimizer_registry.hpp>
#include <string>
#include <unordered_map>
#include <vector>

Model::Model(const std::vector<tf::tensor> &inputs,
             const std::vector<tf::tensor> &outputs)
    : auto_grad_created(false), pred_mapping(false) {
  for (tf::tensor input : inputs)
    this->inputs.push_back(input.getPtr());

  for (tf::tensor output : outputs)
    this->outputs.push_back(output.getPtr());

  std::queue<const Tensor<std::float64_t> *> output_queue;

  for (const Tensor<std::float64_t> *output : this->outputs)
    output_queue.push(output);

  std::vector<const Tensor<std::float64_t> *> validation_inputs =
      getLayersNBackTrackInputs(output_queue);

  for (const Tensor<std::float64_t> *validation_input : validation_inputs)
    if (!std::ranges::contains(this->inputs, validation_input)) {
      LOG(ERROR) << "Fatal! there is a input mismatch.\n";
      break;
    }
  this->doTensorAndLayerMappings();
}

Model::~Model() {
  this->local_training_inputs.clear();
  this->owned_training_inputs.clear();

  this->owned_layers.clear();
  this->owned_losses.clear();
  global_layer_graph.prune();
}

std::shared_ptr<Layer> Model::lockLayer(tf::layer::layer_id id) const {
  auto layer = this->owned_layers.find(id);
  return layer == this->owned_layers.end() ? nullptr : layer->second;
}

std::shared_ptr<Loss> Model::lockLoss(tf::loss::loss_id id) const {
  auto loss = this->owned_losses.find(id);
  return loss == this->owned_losses.end() ? nullptr : loss->second;
}

void Model::resetTrainingGraphState() {
  this->auto_grad_created = false;

  for (tf::layer::layer_id id : this->layers)
    if (std::shared_ptr<Layer> layer = this->lockLayer(id))
      layer->resetGradientRecording();

  for (tf::loss::loss_id id : this->losses)
    if (std::shared_ptr<Loss> loss = this->lockLoss(id))
      loss->resetGradientRecording();
}

void Model::initializeTrainingMappings() {
  for (tf::layer::layer_id id : this->layers)
    this->layer_training_input_mappings[id].assign(
        this->layer_input_mappings[id].size(), nullptr);

  for (tf::loss::loss_id id : this->losses)
    this->loss_training_input_mappings[id].assign(
        loss_input_mappings[id].size(), nullptr);
}

void Model::initializePredictionInputMappings() {
  for (tf::layer::layer_id id : this->layers)
    this->layer_prediction_input_mappings[id].resize(0);
}

std::vector<tf::tensor>
Model::createBatchTargetBuffer(const std::vector<tf::tensor> &training_target) {
  std::vector<tf::tensor> local_temp_outputs(training_target.size());
  unsigned i = 0;
  for (tf::tensor tensor : training_target) {
    std::vector<unsigned> dims;
    for (unsigned it = 0; it < tensor.getNoOfDimensions() - 1; it++)
      dims.push_back(tensor.getDimensions()[it]);
    dims.push_back(this->batch_size);
    local_temp_outputs[i++].tf_create(dims, tensor.dt_type);
  }
  return local_temp_outputs;
}

unsigned Model::selectBatchIndex(int &randIndex, unsigned batch_count) {
  if (this->shuffle_input) {
    randIndex = util::random_engine().rand_unsigned(0, batch_count);
  } else {
    randIndex = (randIndex + 1) % batch_count;
  }
  return static_cast<unsigned>(randIndex);
}

std::unordered_map<std::string, std::float64_t>
Model::getEpochlevelLosses(std::vector<tf::tensor> &predicted_output,
                           const std::vector<tf::tensor> &target_ouput) {

  std::unordered_map<std::string, std::float64_t> temp_losses;
  for (tf::loss::loss_id id : this->losses) {
    std::shared_ptr<Loss> loss = this->lockLoss(id);
    if (!loss)
      continue;
    std::vector<tf::tensor> predicted, target;
    for (unsigned j = 0; j < LossRegistry::instance().noOfInputs(loss); j++) {
      auto it = std::find(this->outputs.begin(), this->outputs.end(),
                          loss_input_mappings[id][j]);
      if (it != outputs.end()) {
        auto index = std::distance(this->outputs.begin(), it);
        predicted.push_back(predicted_output[index].copy());
        target.push_back(target_ouput[index].copy());
      }
    }
    const std::string name = LossRegistry::instance().lossName(loss);
    temp_losses[name] =
        LossRegistry::instance().calculate(loss, predicted, target);
  }
  return temp_losses;
}

void Model::loadTrainingBatch(const std::vector<tf::tensor> &training_inputs,
                              unsigned batch_index) {
  unsigned input_index = 0;
  for (auto local_training_input : this->local_training_inputs) {
    const unsigned element_size = local_training_input->getPtr()->getNoOfElem();
    const unsigned data_offset = batch_index * element_size;
    local_training_input->getPtr()->initPartialData(
        0, element_size,
        training_inputs[input_index].getPtr()->getData() + data_offset);
    input_index++;
  }
}

bool Model::runCallbackBeforeTraining(
    const std::vector<std::shared_ptr<Callback>> &callbacks, tf::model &model) {
  bool flag = true;
  for (const auto &callback : callbacks)
    if (callback)
      flag &= callback->callOnTrainingBegin(model);

  return flag;
}

void Model::runCallbackAfterTraining(
    const std::vector<std::shared_ptr<Callback>> &callbacks) {
  for (const auto &callback : callbacks)
    if (callback)
      callback->callOnTrainingEnd();
}

void Model::runCallbackOnEpochBegin(
    const std::vector<std::shared_ptr<Callback>> &callbacks) {
  for (const auto &callback : callbacks)
    if (callback)
      callback->callOnEpochBegin();
}

void Model::runCallbackOnEpochEnd(
    const std::vector<std::shared_ptr<Callback>> &callbacks) {
  for (const auto &callback : callbacks)
    if (callback)
      callback.get()->callOnEpochEnd();
}

void Model::runCallbackOnBatchBegin(
    const std::vector<std::shared_ptr<Callback>> &callbacks,
    unsigned const epoch_no) {
  for (const auto &callback : callbacks)
    if (callback)
      callback->callOnBatchBegin(epoch_no);
}

void Model::runCallbackOnBatchEnd(
    const std::vector<std::shared_ptr<Callback>> &callbacks,
    unsigned const epoch_no) {
  for (const auto &callback : callbacks)
    if (callback)
      callback->callOnBatchEnd(epoch_no);
}

bool Model::checkEarlyStopping(
    const std::vector<std::shared_ptr<Callback>> &callbacks) {
  bool stop_epoch = false;
  for (const auto &callback : callbacks) {
    if (!callback)
      continue;
    stop_epoch = callback.get()->stopEpoch();
    if (stop_epoch)
      break;
  }

  return stop_epoch;
}

tf::history_container
Model::fit(tf::model &model, const std::vector<tf::tensor> &training_inputs,
           const std::vector<tf::tensor> &training_target,
           const std::vector<tf::tensor> &valdiation_data, unsigned epochs,
           unsigned batch_size,
           std::vector<std::shared_ptr<Callback>> callback_ptr,
           unsigned verbose, bool training) {

  tf::history_container hist;
  if (this->inputs.size() != training_inputs.size()) {
    LOG(ERROR) << "Fatal! # no of input is mismatching with the no of graph "
                  "created during construction.\n";
    return hist;
  }

  this->batch_size = batch_size;
  this->resetTrainingGraphState();
  optimizer.reset_update_graph();
  this->initializeTrainingMappings();

  std::vector<tf::tensor> local_temp_outputs =
      this->createBatchTargetBuffer(training_target);

  this->initilizeInputsForTraining(training_inputs);
  this->setTrainingTensorsForInputLayer();

  unsigned upper_bound =
      training_inputs[0]
          .getPtr()
          ->getDimensions()[training_inputs[0].getPtr()->getNoOfDimensions() -
                            1] /
      this->batch_size; // all training inputs must has same last dimensions
                        // i.e no of input elements, in this case first input
                        // is used
  if (!upper_bound) {
    LOG(ERROR) << "Fatal! number of training samples is smaller than batch "
                  "size.\n";
    return hist;
  }
  int randIndex = -1;

  std::vector<std::shared_ptr<Callback>> callback(callback_ptr);

  if (!this->runCallbackBeforeTraining(callback, model)) {
    LOG(ERROR) << "Fatal! Callback verification failed. Check callback "
                  "configuration\n";
    this->runCallbackAfterTraining(callback);
    this->pred_mapping = false;
    return hist;
  }

  // Training Loop
  {
    /* this sequence is very importent
     * 1. doDummyAndTrainingTensorMapping
     * 2. initialize_gradient()
     * 3. layer backward
     */
    tf::graph_context ctx_compute_n_gradient;

    this->doDummyAndTrainingTensorMapping(); // this requireds to be within
    // graph context
    bool stop_epoch = false;
    for (int i = 0; i < epochs; i++) {
      if (!stop_epoch) {
        this->runCallbackOnEpochBegin(callback);

        for (unsigned batch = 0; batch < upper_bound; batch++) {
          unsigned batch_index = this->selectBatchIndex(randIndex, upper_bound);
          this->loadTrainingBatch(training_inputs, batch_index);

          for (tf::layer::layer_id id : this->layers)
            if (std::shared_ptr<Layer> layer = this->lockLayer(id))
              layer->initializeParameters();

          /* making loss graph */
          this->setTargetOutputForLoss(training_target, local_temp_outputs,
                                       batch_index);

          this->runCallbackOnBatchBegin(callback, i);

          ctx_compute_n_gradient.run(); // forward propagation
          if (training) {
            if (!this->auto_grad_created) {
              ctx_compute_n_gradient.initialize_gradient();

              for (tf::layer::layer_id id : this->layers)
                if (std::shared_ptr<Layer> layer = this->lockLayer(id))
                  layer->backward(optimizer);

              for (tf::loss::loss_id id : this->losses)
                if (std::shared_ptr<Loss> loss = this->lockLoss(id))
                  LossRegistry::instance().backward(loss);

              this->auto_grad_created = true;
            }
            ctx_compute_n_gradient.compute_gradient(); // back propagation
            optimizer.execute();
          }

          this->runCallbackOnBatchEnd(callback, i);
          hist.record_history_on_batch_end(i, losses);
        } // end of batch

        for (tf::layer::layer_id id : this->layers)
          if (std::shared_ptr<Layer> layer = this->lockLayer(id))
            layer->initializeParameters();

        std::vector<tf::tensor> epoch_level_predicted_output =
            this->predict(training_inputs);
        if (!this->pred_mapping)
          this->pred_mapping = true;

        std::unordered_map<std::string, std::float64_t> epoch_level_losses =
            this->getEpochlevelLosses(epoch_level_predicted_output,
                                      training_target);

        this->runCallbackOnEpochEnd(callback);
        hist.record_history_on_epoch_end(epoch_level_losses);
        stop_epoch = this->checkEarlyStopping(callback);
      } else {
        LOG(ERROR)
            << "Early stopping has been triggered, Stopping the epoch at " << i
            << "\n";
        break;
      }
    } // end of epoch
    this->runCallbackAfterTraining(callback);
    this->pred_mapping = false;
  }
  return hist;
}

std::vector<tf::tensor> Model::predict(std::vector<tf::tensor> inputs) {
  if (!pred_mapping) {
    prediction_outputs.resize(outputs.size());
    initializePredictionInputMappings();
    initializeInputsForPrediction(inputs);
  }
  doLayerCalculation();
  return prediction_outputs;
}

/** this subroutine inspects each input and creates a local training-input which
 * has exact dimention of each incoming input tensor except the n-th dim which
 * is tranculated to batch size*/
void Model::initilizeInputsForTraining(
    const std::vector<tf::tensor> &incoming_training_inputs) {
  // Drop the previous fit's batch buffers before creating new mappings.
  this->local_training_inputs.clear();
  this->owned_training_inputs.clear();

  for (const tf::tensor &input : incoming_training_inputs) {
    std::vector<unsigned> dims;
    for (unsigned i = 0; i < input.getPtr()->getNoOfDimensions() - 1; i++)
      dims.push_back(input.getPtr()->getDimensions()[i]);

    dims.push_back(this->batch_size);
    auto buffer = std::make_unique<tf::tensor>();
    buffer->tf_create(dims, tf_float64);
    this->local_training_inputs.push_back(buffer.get());
    this->owned_training_inputs.push_back(std::move(buffer));
  }
}

void Model::initializeInputsForPrediction(
    const std::vector<tf::tensor> &prediction_inputs) {

  for (tf::layer::layer_id id : this->layers) {
    this->layer_prediction_input_mappings[id].resize(
        layer_input_mappings[id].size());

    unsigned i = 0;
    for (Tensor<std::float64_t> *input : this->inputs) {
      auto it = std::find(layer_input_mappings[id].begin(),
                          layer_input_mappings[id].end(), input);
      if (it != layer_input_mappings[id].end()) {
        unsigned index =
            std::distance(layer_input_mappings[id].begin(), it);

        this->layer_prediction_input_mappings[id][index] = prediction_inputs[i];
      }
      i++;
    }
  }
}

/** this subroutine
 * 1. first accumulate all the layers from global layer graph
 * 2. then back tracks each layer with its output to reach input then at the end
 * the child nodes are kept in a vector to validate the inputs */
std::vector<const Tensor<std::float64_t> *> Model::getLayersNBackTrackInputs(
    std::queue<const Tensor<std::float64_t> *> &output_queue) {

  std::vector<const Tensor<std::float64_t> *> validation_inputs;
  std::unordered_set<tf::layer::layer_id> layer_set;

  while (output_queue.size()) {
    const Tensor<std::float64_t> *output_tensor = output_queue.front();
    output_queue.pop();

    const tf::layer::layer_id id =
        global_layer_graph.getLayerOfOutgoingTensor(output_tensor);

    if (id == tf::layer::invalid_layer_id) {
      validation_inputs.push_back(output_tensor);
      continue;
    }

    if (layer_set.count(id)) {
      LOG(ERROR) << "Fatal! A tensor is not connected to any layer.\n";
      continue;
    }

    std::shared_ptr<Layer> layer = global_layer_graph.lock(id);
    if (!layer) {
      validation_inputs.push_back(output_tensor);
      continue;
    }

    layer_set.insert(id);
    this->layers.push_back(id);
    this->owned_layers.insert_or_assign(id, layer);
    layer_input_mappings[id] = layer->getInputTensors();

    for (tf::tensor *output : layer->getOutputTensors())
      layer_output_mappings[id].push_back(output);

    for (const Tensor<std::float64_t> *input : layer_input_mappings[id])
      output_queue.push(input);
  }
  return validation_inputs;
}

/** this subroutine at creates a unordered map of layer-&-tensor where it
 * identifies the layers associated with inputs (i.e input layers) then creates
 * a layer-tensor map */
void Model::doTensorAndLayerMappings() {
  for (const Tensor<std::float64_t> *input : this->inputs) {
    const std::vector<tf::layer::layer_id> input_layer_ids =
        global_layer_graph.getLayersOfIncomingTensor(input);
    for (tf::layer::layer_id id : this->layers)
      if (std::ranges::contains(input_layer_ids, id)) {
        this->input_layer_mappings[input].push_back(id);
        this->input_layers.push_back(id);
      }
  }
}

/** At fit: this subroutine primarily does
 * 1. feed each layer with it's proper input and calls forward().
 * 2 forward give output vector for the layer down stream.
 * 3. based the output it feed the ouput tensor to each layer's input and then
 * call forward() on them recursively. */
void Model::doDummyAndTrainingTensorMapping() {
  std::queue<tf::layer::layer_id> training_layers;
  bool flag;

  for (tf::layer::layer_id id : this->input_layers)
    training_layers.push(id);

  while (training_layers.size()) {
    const tf::layer::layer_id id = training_layers.front();
    training_layers.pop();

    std::shared_ptr<Layer> this_layer = this->lockLayer(id);
    if (!this_layer) {
      LOG(ERROR) << "Fatal! A model layer is no longer available.\n";
      continue;
    }

    /** if all the inputs are ready for a layer(i.e !nullptr) call forward on
     * that layer else, push it back on the queue for subsequent layers to give
     * its output
     */
    if (!std::ranges::contains(this->layer_training_input_mappings[id],
                               nullptr)) {
      const std::vector<tf::tensor *> &this_layer_training_outputs =
          this_layer->forward(this->layer_training_input_mappings[id],
                              this->batch_size);

      /* now find where each output is going*/
      for (tf::layer::layer_id downstream_id : this->layers) {
        flag = false;
        unsigned i = 0;
        for (const tf::tensor *this_layer_output :
             this->layer_output_mappings[id]) {
          auto it = std::find(
              this->layer_input_mappings[downstream_id].begin(),
              this->layer_input_mappings[downstream_id].end(),
              this_layer_output->getPtr());

          if (it != this->layer_input_mappings[downstream_id].end()) {
            unsigned index = std::distance(
                this->layer_input_mappings[downstream_id].begin(), it);
            this->layer_training_input_mappings[downstream_id][index] =
                this_layer_training_outputs[i++];
            flag = true;
          }
        }
        if (flag)
          training_layers.push(downstream_id);
      }
      /* map training layer output with loss training input */
      for (tf::loss::loss_id loss_id : this->losses) {
        unsigned i = 0;
        for (const tf::tensor *this_layer_output :
             this->layer_output_mappings[id]) {
          auto it = std::find(this->loss_input_mappings[loss_id].begin(),
                              this->loss_input_mappings[loss_id].end(),
                              this_layer_output->getPtr());

          if (it != this->loss_input_mappings[loss_id].end()) {
            unsigned index =
                std::distance(this->loss_input_mappings[loss_id].begin(), it);
            this->loss_training_input_mappings[loss_id][index] =
                this_layer_training_outputs[i++];
          }
        }
      }
    } else {
      training_layers.push(id);
    }
  }

  /* Call forward on loss to create forward graph on loss.*/
  for (tf::loss::loss_id id : this->losses) {
    std::shared_ptr<Loss> loss = this->lockLoss(id);
    if (!loss)
      continue;
    if (!std::ranges::contains(this->loss_training_input_mappings[id], nullptr))
      LossRegistry::instance().forward(
          loss, this->loss_training_input_mappings[id], this->batch_size);
    else {
      LOG(ERROR) << "Fatal! a loss is not properly mapped with layer output.";
      throw std::runtime_error(
          "Exiting due to mismatch with loss and layer output.");
    }
  }
}

void Model::doLayerCalculation() {
  std::queue<tf::layer::layer_id> prediction_layers;

  bool flag;
  for (tf::layer::layer_id id : this->input_layers)
    prediction_layers.push(id);

  while (prediction_layers.size()) {
    const tf::layer::layer_id id = prediction_layers.front();
    prediction_layers.pop();

    std::shared_ptr<Layer> this_layer = this->lockLayer(id);
    if (!this_layer) {
      LOG(ERROR) << "Fatal! A model layer is no longer available.\n";
      continue;
    }

    /** if all the inputs are ready for a layer(i.e !nullptr) call forward on
     * that layer else, push it back on the queue for subsequent layers to give
     * its output
     */
    if (this->layer_prediction_input_mappings[id].size() ==
        layer_input_mappings[id].size()) {
      std::vector<tf::tensor> this_layer_prediction_outputs =
          this_layer->calculate(this->layer_prediction_input_mappings[id]);

      /* now find where each output is going*/
      for (tf::layer::layer_id downstream_id : this->layers) {
        flag = false;
        unsigned i = 0;
        for (const tf::tensor *this_layer_output :
             this->layer_output_mappings[id]) {
          auto it = std::find(
              this->layer_input_mappings[downstream_id].begin(),
              this->layer_input_mappings[downstream_id].end(),
              this_layer_output->getPtr());

          if (it != this->layer_input_mappings[downstream_id].end()) {
            unsigned index = std::distance(
                this->layer_input_mappings[downstream_id].begin(), it);
            this->layer_prediction_input_mappings[downstream_id][index] =
                this_layer_prediction_outputs[index].copy();
            flag = true;
          }
        }
        if (flag)
          prediction_layers.push(downstream_id);
      }
      // place outputs to its place
      for (const tf::tensor *layer_output : layer_output_mappings[id]) {
        auto it = std::find(this->outputs.begin(), this->outputs.end(),
                            layer_output->getPtr());

        if (it != this->outputs.end()) {
          unsigned index = std::distance(this->outputs.begin(), it);
          this->prediction_outputs[index] =
              this_layer_prediction_outputs[index].copy();
        }
      }

    } else {
      prediction_layers.push(id);
    }
  }
}

/** this subroutine setup training-tensors for each input layer based on the
 * relative position of input tensor feed during the model construction */
void Model::setTrainingTensorsForInputLayer() {
  for (tf::layer::layer_id id : this->input_layers) {
    int i = 0;
    for (const Tensor<std::float64_t> *input_tensor :
         this->layer_input_mappings[id]) {
      auto it =
          std::find(this->inputs.begin(), this->inputs.end(), input_tensor);
      if (it != this->inputs.end()) {
        unsigned index = std::distance(this->inputs.begin(), it);
        this->layer_training_input_mappings[id][i] =
            this->local_training_inputs[index];
      }
      i++;
    }
  }
}

void Model::shuffle(bool shuffle) { this->shuffle_input = shuffle; }

void Model::compile(std::string_view optimizer_type,
                    std::string_view loss_type) {
  const std::string optimizer_name(optimizer_type);
  if (!OptimizerRegistry::instance().contains(optimizer_name)) {
    LOG(ERROR) << "Unknown optimizer type '" << optimizer_name << "'.\n";
    throw std::invalid_argument("Unknown optimizer type: " + optimizer_name);
  }

  this->compile(tf::optimizer::optimizer(optimizer_type), loss_type);
}

void Model::compile(const tf::optimizer::optimizer &optimizer,
                    std::string_view loss_type) {
  if (!optimizer.valid())
    throw std::invalid_argument("Invalid optimizer handle");

  const std::string type_name(loss_type);
  if (!LossRegistry::instance().contains(type_name)) {
    LOG(ERROR) << "Unknown loss type '" << type_name << "'.\n";
    throw std::invalid_argument("Unknown loss type: " + type_name);
  }

  // Keep the historical auto-generated names for the built-in losses.
  std::string base_name = type_name;
  if (loss_type == tf::loss::type::squared_error)
    base_name = "mean_squared_loss";

  // Build the replacement state first so a failed compile leaves the old
  // optimizer and losses intact.
  std::vector<tf::loss::loss_id> new_losses;
  std::unordered_map<tf::loss::loss_id, std::shared_ptr<Loss>> new_owned_losses;
  std::unordered_map<tf::loss::loss_id, std::vector<Tensor<std::float64_t> *>>
      new_loss_input_mappings;

  for (unsigned i = 0; i < outputs.size(); i++) {
    LossRegistry::LossHandle loss = LossRegistry::instance().create(
        type_name,
        makeLossConfig({outputs[i]}, base_name + "_" + std::to_string(i + 1)));
    if (!loss)
      throw std::runtime_error("Failed to create loss: " + type_name);

    const tf::loss::loss_id id = LossRegistry::instance().idOf(loss);
    new_owned_losses[id] = std::move(loss);
    new_losses.push_back(id);
    new_loss_input_mappings[id].push_back(outputs[i]);
  }

  // Release the previous optimizer/losses before committing the new ones.
  this->optimizer = {};
  this->owned_losses.clear();
  this->losses.clear();
  this->loss_input_mappings.clear();
  this->loss_training_input_mappings.clear();

  this->optimizer = optimizer;
  this->owned_losses = std::move(new_owned_losses);
  this->losses = std::move(new_losses);
  this->loss_input_mappings = std::move(new_loss_input_mappings);

  this->resetTrainingGraphState();
  optimizer.reset_update_graph();
}

tf::loss::loss Model::getModelLoss(Tensor<std::float64_t> *output_tensor) {
  for (tf::loss::loss_id id : losses) {
    if (!std::ranges::contains(this->loss_input_mappings[id], output_tensor))
      continue;
    if (std::shared_ptr<Loss> loss = this->lockLoss(id))
      return tf::loss::loss(std::move(loss));
  }
  LOG(ERROR) << "Fatal! Can't retrive the loss for the given output.\n";
  throw std::runtime_error("Exiting due to prior violation.\n");
}

void Model::setTargetOutputForLoss(
    const std::vector<tf::tensor> &training_target,
    std::vector<tf::tensor> &local_temp_outputs, unsigned data_index) {

  for (tf::loss::loss_id id : this->losses) {
    std::shared_ptr<Loss> loss = this->lockLoss(id);
    if (!loss)
      continue;
    std::vector<tf::tensor> temp_tensor(this->loss_input_mappings[id].size());
    unsigned index = 0;
    for (Tensor<std::float64_t> *tensor : this->loss_input_mappings[id]) {
      auto it = std::find(this->outputs.begin(), this->outputs.end(), tensor);
      unsigned index = 0;
      if (it != this->outputs.end()) {
        index = std::distance(this->outputs.begin(), it);

        std::vector<unsigned> dims;
        unsigned stride = 1;
        for (unsigned j = 0; j < local_temp_outputs[index].getNoOfDimensions();
             j++) {
          dims.push_back(local_temp_outputs[index].getDimensions()[j]);
          stride *= local_temp_outputs[index].getDimensions()[j];
        }
        temp_tensor[index].tf_create(dims, training_target[index].dt_type);
        temp_tensor[index].tensor_of(training_target[index].getData() +
                                     stride * data_index);
      }
      LossRegistry::instance().setTargetOutput(loss, temp_tensor);
    }
  }
}