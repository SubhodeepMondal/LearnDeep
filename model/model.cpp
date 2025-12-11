// C++ Headers
#include <algorithm>
#include <iterator>
#include <queue>

// Library Headers
#include "model.hpp"
#include <core/graph/graph_context.hpp>
#include <model/model.hpp>

Model::Model(const std::vector<Tensor<std::float64_t> *> &inputs,
             const std::vector<Tensor<std::float64_t> *> &outputs) {
  this->inputs = inputs;
  this->outputs = outputs;
  std::queue<Tensor<std::float64_t> *> output_queue;

  for (Tensor<std::float64_t> *output : outputs)
    output_queue.push(output);

  std::vector<Tensor<std::float64_t> *> validation_inputs =
      getLayers(output_queue);

  for (Tensor<std::float64_t> *validation_input : validation_inputs)
    if (!std::ranges::contains(this->inputs, validation_input)) {
      LOG(ERROR) << "Fatal! there is a input mismatch.\n";
      break;
    }
  this->doTensorAndLayerMappings();
}

void Model::fit(std::vector<Tensor<std::float64_t> *> training_inputs,
                std::vector<Tensor<std::float64_t> *> training_target,
                std::vector<Tensor<std::float64_t> *> valdiation_data,
                unsigned epochs, unsigned batch_size, unsigned callback,
                unsigned verbose) {
  if (this->inputs.size() == training_inputs.size()) {

    for (Layer *layer : this->layers)
      this->layer_training_input_mappings[layer].assign(
          this->layer_input_mappings[layer].size(), nullptr);

    this->getTrainingTensorsForInputLayer(training_inputs);

    // Training Loop
    {
      GraphContext ctx_compute_n_gradient;

      this->doDummyAndTrainingTensorMapping(training_inputs);

      ctx_compute_n_gradient.graph_initilize_gradient();

      for (int i = 0; i < epochs; i++) {
        ctx_compute_n_gradient.run();                    // forward propagation
        ctx_compute_n_gradient.graph_compute_gradeint(); // back propagation
      }
    }
  } else {
    LOG(ERROR) << "Fatal! # no of input is mismatching with the no of graph "
                  "created during construction.\n";
  }
}

std::vector<Tensor<std::float64_t> *>
Model::getLayers(std::queue<Tensor<std::float64_t> *> &output_queue) {

  std::vector<Tensor<std::float64_t> *> validation_inputs;
  std::unordered_set<Layer *> layer_set;
  Layer *prev_layer = nullptr;

  while (output_queue.size()) {
    Tensor<std::float64_t> *output_tensor = output_queue.front();
    output_queue.pop();

    Layer *this_layer =
        global_layer_graph.getLayerOfOutgoingTensor(output_tensor);

    if (this_layer && !layer_set.count(this_layer)) {
      layer_set.insert(this_layer);
      this->layers.push_back(this_layer);
      layer_input_mappings[this_layer] = this_layer->getInputTensors();
      layer_output_mappings[this_layer] = this_layer->getOutputTensors();
      for (Tensor<std::float64_t> *incoming_tensor :
           layer_input_mappings[this_layer])
        output_queue.push(incoming_tensor);
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
  for (Tensor<std::float64_t> *input : this->inputs) {
    std::vector<Layer *> input_layers =
        global_layer_graph.getLayersOfIncomingTensor(input);
    for (Layer *layer : this->layers)
      if (std::ranges::contains(input_layers, layer)) {
        this->input_layer_mappings[input].push_back(layer);
        this->input_layers.push_back(layer);
      }
  }
}

/** At fit this subroutine primarily does
 * 1. feed each layer with it's proper input and calls forward()
 * 2 forward give output vector for the layer
 * 3. based the output it feed the ouput tensor to each layer's input and then
 * call forward on them */
void Model::doDummyAndTrainingTensorMapping(
    const std::vector<Tensor<std::float64_t> *> &training_inputs) {
  std::queue<Layer *> training_layers;
  bool flag;

  for (Layer *layer : this->input_layers)
    training_layers.push(layer);

  while (training_layers.size()) {
    Layer *this_layer = training_layers.front();
    training_layers.pop();

    /** if all the inputs are ready(i.e !nullptr) call forward on the layer
     * else, push it back on the queue for subsequent layers to give its output
     */
    if (!std::ranges::contains(this->layer_training_input_mappings[this_layer],
                               nullptr)) {
      std::vector<Tensor<std::float64_t> *> this_layer_training_output =
          this_layer->forward(this->layer_training_input_mappings[this_layer]);

      /* now find where each output is going*/
      for (Layer *layer : this->layers) {
        flag = false;
        for (Tensor<std::float64_t> *this_layer_output :
             this->layer_output_mappings[this_layer]) {
          auto it = std::find(this->layer_input_mappings[layer].begin(),
                              this->layer_input_mappings[layer].end(),
                              this_layer_output);

          if (it != this->layer_input_mappings[layer].end()) {
            unsigned index =
                std::distance(this->layer_input_mappings[layer].begin(), it);
            this->layer_training_input_mappings[layer][index] =
                this_layer_output;
            flag = true;
          }
        }
        if (flag)
          training_layers.push(layer);
      }

    } else {
      training_layers.push(this_layer);
    }
  }
}

void Model::getTrainingTensorsForInputLayer(
    const std::vector<Tensor<std::float64_t> *> &training_inputs) {
  for (Layer *layer : this->input_layers) {
    int i = 0;
    for (Tensor<std::float64_t> *input_tensor :
         this->layer_input_mappings[layer]) {
      auto it =
          std::find(this->inputs.begin(), this->inputs.end(), input_tensor);
      if (it != this->inputs.end()) {
        unsigned index = std::distance(this->inputs.begin(), it);
        this->layer_training_input_mappings[layer][i] = training_inputs[index];
      }
      i++;
    }
  }
}