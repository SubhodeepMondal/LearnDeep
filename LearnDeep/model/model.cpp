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
  for (auto local_training_input : local_training_inputs)
    delete local_training_input;

  this->owned_layers.clear();
  global_layer_graph.prune();
}

std::shared_ptr<Layer> Model::lockLayer(tf::layer::layer_id id) const {
  auto layer = this->owned_layers.find(id);
  return layer == this->owned_layers.end() ? nullptr : layer->second;
}

void Model::initializeTrainingMappings() {
  for (tf::layer::layer_id id : this->layers)
    this->layer_training_input_mappings[id].assign(
        this->layer_input_mappings[id].size(), nullptr);

  for (tf::loss::loss *loss : this->losses)
    this->loss_training_input_mappings[loss].assign(
        loss_input_mappings[loss].size(), nullptr);
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
  for (tf::loss::loss *loss : this->losses) {
    std::vector<tf::tensor> predicted, target;
    for (unsigned j = 0; j < loss->get_no_of_inputs(); j++) {
      auto it = std::find(this->outputs.begin(), this->outputs.end(),
                          loss_input_mappings[loss][j]);
      if (it != outputs.end()) {
        auto index = std::distance(this->outputs.begin(), it);
        predicted.push_back(predicted_output[index].copy());
        target.push_back(target_ouput[index].copy());
      }
    }
    temp_losses[loss->get_loss_name()] = loss->calculate(predicted, target);
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
                  layer->backward(optimizer.getPtr());

              for (tf::loss::loss *loss : this->losses)
                loss->backward();

              this->auto_grad_created = true;
            }
            ctx_compute_n_gradient.compute_gradient(); // back propagation
            optimizer.execute_optimizer();
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
  for (const tf::tensor &input : incoming_training_inputs) {
    std::vector<unsigned> dims;
    for (unsigned i = 0; i < input.getPtr()->getNoOfDimensions() - 1; i++)
      dims.push_back(input.getPtr()->getDimensions()[i]);

    dims.push_back(this->batch_size);
    temp_training_input_tensor = new tf::tensor();
    temp_training_input_tensor->tf_create(dims, tf_float64);
    this->local_training_inputs.push_back(temp_training_input_tensor);
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
      for (tf::loss::loss *loss : this->losses) {
        unsigned i = 0;
        for (const tf::tensor *this_layer_output :
             this->layer_output_mappings[id]) {
          auto it = std::find(this->loss_input_mappings[loss].begin(),
                              this->loss_input_mappings[loss].end(),
                              this_layer_output->getPtr());

          if (it != this->loss_input_mappings[loss].end()) {
            unsigned index =
                std::distance(this->loss_input_mappings[loss].begin(), it);
            this->loss_training_input_mappings[loss][index] =
                this_layer_training_outputs[i++];
          }
        }
      }
    } else {
      training_layers.push(id);
    }
  }

  /* Call forward on loss to create forward graph on loss.*/
  for (tf::loss::loss *loss : this->losses) {
    if (!std::ranges::contains(this->loss_training_input_mappings[loss],
                               nullptr))
      loss->forward(this->loss_training_input_mappings[loss], this->batch_size);
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

void Model::compile(const OptimizerType optimizer_type,
                    const LossType loss_type) {

  this->optimizer = tf::optimizer(optimizer_type);
  int mse_count, cce_count;
  mse_count = cce_count = 0;
  tf::loss::loss *temp_loss = nullptr;
  for (unsigned i = 0; i < outputs.size(); i++) {
    switch (loss_type) {
    case (LossType::squared_error): {
      mse_count++;
      temp_loss =
          new tf::loss::loss(loss_type, {outputs[i]},
                       "mean_squared_loss_" + std::to_string(mse_count));
      break;
    }
    case (LossType::categorical_cross_entropy): {
      cce_count++;
      temp_loss = new tf::loss::loss(loss_type, {outputs[i]},
                               "categorical_cross_entropy_" +
                                   std::to_string(cce_count));
      break;
    }
    default: {
      temp_loss = nullptr;
      break;
    }
    }
    this->losses.push_back(temp_loss);
    loss_input_mappings[temp_loss].push_back(outputs[i]);
  }
}

tf::loss::loss Model::getModelLoss(Tensor<std::float64_t> *output_tensor) {
  tf::loss::loss *loss = nullptr;
  for (tf::loss::loss *temp_loss : losses) {
    if (std::ranges::contains(this->loss_input_mappings[temp_loss],
                              output_tensor))
      loss = temp_loss;
  }
  if (loss)
    return *loss;
  else {
    LOG(ERROR) << "Fatal! Can't retrive the loss for the given output.\n";
    throw std::runtime_error("Exiting due to prior violation.\n");
  }
}

void Model::setTargetOutputForLoss(
    const std::vector<tf::tensor> &training_target,
    std::vector<tf::tensor> &local_temp_outputs, unsigned data_index) {

  for (tf::loss::loss *loss : this->losses) {
    std::vector<tf::tensor> temp_tensor(this->loss_input_mappings[loss].size());
    unsigned index = 0;
    for (Tensor<std::float64_t> *tensor : this->loss_input_mappings[loss]) {
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
      loss->set_target_output(temp_tensor);
    }
  }
}