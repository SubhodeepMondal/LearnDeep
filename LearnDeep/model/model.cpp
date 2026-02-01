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

Model::Model(const std::vector<tf::tensor> &inputs,
             const std::vector<tf::tensor> &outputs) {
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
}

void Model::fit(const std::vector<tf::tensor> &training_inputs,
                const std::vector<tf::tensor> &training_target,
                const std::vector<tf::tensor> &valdiation_data, unsigned epochs,
                unsigned batch_size, Callback *callback, unsigned verbose) {
  if (this->inputs.size() == training_inputs.size()) {

    for (Layer *layer : this->layers)
      this->layer_training_input_mappings[layer].assign(
          this->layer_input_mappings[layer].size(), nullptr);

    for (tf::loss *loss : this->losses)
      this->loss_training_input_mappings[loss].assign(
          loss_input_mappings[loss].size(), nullptr);

    std::vector<tf::tensor> local_temp_outputs(training_target.size());
    unsigned i = 0;
    for (tf::tensor tensor : training_target) {
      std::vector<unsigned> dims;
      for (unsigned it = 0; it < tensor.getNoOfDimensions() - 1; it++)
        dims.push_back(tensor.getDimensions()[it]);
      dims.push_back(batch_size);

      local_temp_outputs[i++].tf_create(dims, tensor.dt_type);
    }

    this->batch_size = batch_size;
    this->initilizeInputsForTraining(training_inputs);

    this->setTrainingTensorsForInputLayer();
    int randIndex = -1;
    unsigned lower_bound = 0;
    unsigned upper_bound =
        training_inputs[0]
            .getPtr()
            ->getDimensions()[training_inputs[0].getPtr()->getNoOfDimensions() -
                              1] /
        this->batch_size; // all training inputs must has same last dimensions
                          // i.e no of input elements, in this case first input
                          // is used

    unsigned element_size;

    // Training Loop
    {
      tf::graph_context ctx_compute_n_gradient;

      this->doDummyAndTrainingTensorMapping(); // this requireds to be within
                                               // graph context

      ctx_compute_n_gradient.initialize_gradient();

      for (int i = 0; i < epochs; i++) {
        if (this->shuffle_input) {
          randIndex =
              util::random_engine().rand_unsigned(lower_bound, upper_bound);
        } else {
          randIndex++;
        }
        unsigned it = 0;
        for (auto local_training_input : this->local_training_inputs) {
          element_size = local_training_input->getPtr()->getNoOfElem();
          unsigned index = randIndex * element_size;
          local_training_input->getPtr()->initPartialData(
              0, element_size, training_inputs[it].getPtr()->getData() + index);
          it++;
        }

        for (tf::loss *loss : this->losses) {
          std::vector<tf::tensor> temp_tensor(
              this->loss_input_mappings[loss].size());
          unsigned index = 0;
          for (Tensor<std::float64_t> *tensor :
               this->loss_input_mappings[loss]) {
            auto it =
                std::find(this->outputs.begin(), this->outputs.end(), tensor);
            unsigned index = 0;
            if (it != this->outputs.end()) {
              index = std::distance(this->outputs.begin(), it);

              std::vector<unsigned> dims;
              unsigned stride = 1;
              for (unsigned j = 0;
                   j < local_temp_outputs[index].getNoOfDimensions(); j++) {
                dims.push_back(local_temp_outputs[index].getDimensions()[j]);
                if (j != local_temp_outputs[index].getNoOfDimensions() - 1)
                  stride *= local_temp_outputs[index].getDimensions()[j];
              }
              temp_tensor[index].tf_create(dims,
                                           training_target[index].dt_type);
              temp_tensor[index].tensor_of(training_target[index].getData() +
                                           stride * randIndex);
            }
            loss->set_target_output(temp_tensor);
          }
        }

        callback->callOnEpochBegin();

        ctx_compute_n_gradient.run();              // forward propagation
        ctx_compute_n_gradient.compute_gradient(); // back propagation

        callback->callOnEpochEnd();
        callback->recordLossOnEpochEnd();
      }
    }
  } else {
    LOG(ERROR) << "Fatal! # no of input is mismatching with the no of graph "
                  "created during construction.\n";
  }
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

/** this subroutine
 * 1. first accumulate all the layers from global layer graph
 * 2. then back tracks each layer with its output to reach input then at the end
 * the child nodes are kept in a vector to validate the inputs */
std::vector<const Tensor<std::float64_t> *> Model::getLayersNBackTrackInputs(
    std::queue<const Tensor<std::float64_t> *> &output_queue) {

  std::vector<const Tensor<std::float64_t> *> validation_inputs;
  std::unordered_set<Layer *> layer_set;
  Layer *prev_layer = nullptr;

  while (output_queue.size()) {
    const Tensor<std::float64_t> *output_tensor = output_queue.front();
    output_queue.pop();

    Layer *this_layer =
        global_layer_graph.getLayerOfOutgoingTensor(output_tensor);

    if (this_layer && !layer_set.count(this_layer)) {
      layer_set.insert(this_layer);
      this->layers.push_back(this_layer);
      layer_input_mappings[this_layer] = this_layer->getInputTensors();

      for (unsigned i = 0; i < this_layer->getOutputTensors().size(); i++)
        layer_output_mappings[this_layer].push_back(
            this_layer->getOutputTensors()[i]);

      for (unsigned i = 0; i < layer_input_mappings[this_layer].size(); i++)
        output_queue.push(layer_input_mappings[this_layer][i]);

    } else if (!this_layer) {
      validation_inputs.push_back(output_tensor);
    } else {
      LOG(ERROR) << "Fatal! A tensor is not connected to any layer.\n";
    }
  }
  return validation_inputs;
}

void Model::initializeLayerGraph() {}
/** this subroutine at creates a unordered map of layer-&-tensor where it
 * identifies the layers associated with inputs (i.e input layers) then creates
 * a layer-tensor map */
void Model::doTensorAndLayerMappings() {
  for (const Tensor<std::float64_t> *input : this->inputs) {
    std::vector<Layer *> input_layers =
        global_layer_graph.getLayersOfIncomingTensor(input);
    for (Layer *layer : this->layers)
      if (std::ranges::contains(input_layers, layer)) {
        this->input_layer_mappings[input].push_back(layer);
        this->input_layers.push_back(layer);
      }
  }
}

/** At fit: this subroutine primarily does
 * 1. feed each layer with it's proper input and calls forward().
 * 2 forward give output vector for the layer down stream.
 * 3. based the output it feed the ouput tensor to each layer's input and then
 * call forward() on them recursively. */
void Model::doDummyAndTrainingTensorMapping() {
  std::queue<Layer *> training_layers;
  std::queue<tf::loss *> training_losses;
  bool flag;

  for (Layer *layer : this->input_layers)
    training_layers.push(layer);

  while (training_layers.size()) {
    Layer *this_layer = training_layers.front();
    training_layers.pop();

    /** if all the inputs are ready for a layer(i.e !nullptr) call forward on
     * that layer else, push it back on the queue for subsequent layers to give
     * its output
     */
    if (!std::ranges::contains(this->layer_training_input_mappings[this_layer],
                               nullptr)) {
      const std::vector<tf::tensor *> &this_layer_training_outputs =
          this_layer->forward(this->layer_training_input_mappings[this_layer],
                              this->batch_size);

      /* now find where each output is going*/
      for (Layer *layer : this->layers) {
        flag = false;
        unsigned i = 0;
        for (const tf::tensor *this_layer_output :
             this->layer_output_mappings[this_layer]) {
          auto it = std::find(this->layer_input_mappings[layer].begin(),
                              this->layer_input_mappings[layer].end(),
                              this_layer_output->getPtr());

          if (it != this->layer_input_mappings[layer].end()) {
            unsigned index =
                std::distance(this->layer_input_mappings[layer].begin(), it);
            this->layer_training_input_mappings[layer][index] =
                this_layer_training_outputs[i++];
            flag = true;
          }
        }
        if (flag)
          training_layers.push(layer);
      }
      /* map training layer output with loss training input */
      for (tf::loss *loss : this->losses) {
        unsigned i = 0;
        for (const tf::tensor *this_layer_output :
             this->layer_output_mappings[this_layer]) {
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
      training_layers.push(this_layer);
    }
  }

  /* Call forward on loss to create forward graph on loss.*/
  for (tf::loss *loss : this->losses) {
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

/** this subroutine setup training-tensors for each input layer based on the
 * relative position of input tensor feed during the model construction */
void Model::setTrainingTensorsForInputLayer() {
  for (Layer *layer : this->input_layers) {
    int i = 0;
    for (const Tensor<std::float64_t> *input_tensor :
         this->layer_input_mappings[layer]) {
      auto it =
          std::find(this->inputs.begin(), this->inputs.end(), input_tensor);
      if (it != this->inputs.end()) {
        unsigned index = std::distance(this->inputs.begin(), it);
        this->layer_training_input_mappings[layer][i] =
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
  for (unsigned i = 0; i < outputs.size(); i++) {
    tf::loss *temp_loss = new tf::loss(loss_type, {outputs[i]});
    this->losses.push_back(temp_loss);
    loss_input_mappings[temp_loss].push_back(outputs[i]);
  }
  // this->metric = tf::metric(metric_type);
}

tf::loss Model::getModelLoss(Tensor<std::float64_t> *output_tensor) {
  tf::loss *loss = nullptr;
  for (tf::loss *temp_loss : losses) {
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