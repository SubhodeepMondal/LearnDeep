// C++ Headers
#include <algorithm>
#include <iterator>
#include <queue>

// Library Headers
#include "model.hpp"
#include <core/graph/graph_context.hpp>
#include <core/utility/initializers.hpp>
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
                unsigned epochs, unsigned batch_size, Callback *callback,
                unsigned verbose) {
  if (this->inputs.size() == training_inputs.size()) {

    for (Layer *layer : this->layers)
      this->layer_training_input_mappings[layer].assign(
          this->layer_input_mappings[layer].size(), nullptr);

    this->batch_size = batch_size;
    this->initilizeInputsForTraining(training_inputs);

    this->getTrainingTensorsForInputLayer();
    int randIndex = -1;
    unsigned lower_bound = 0;
    unsigned upper_bound =
        training_inputs[0]
            ->getDimensions()[training_inputs[0]->getNoOfDimensions() - 1] /
        this->batch_size; // all training inputs must has same last dimensions
                          // i.e no of input elements, in this case first input
                          // is used

    unsigned element_size;

    // Training Loop
    {
      GraphContext ctx_compute_n_gradient;

      this->doDummyAndTrainingTensorMapping(local_training_inputs);

      ctx_compute_n_gradient.graph_initilize_gradient();

      for (int i = 0; i < epochs; i++) {
        if (this->shuffle_input) {
          randIndex =
              util::random_engine().rand_unsigned(lower_bound, upper_bound);
        } else {
          randIndex++;
        }
        unsigned it = 0;
        for (auto local_training_input : local_training_inputs) {
          element_size = local_training_input->getNoOfElem();
          unsigned index = randIndex * element_size;
          local_training_input->initPartialData(
              0, element_size, training_inputs[it]->getData() + index);
          it++;
        }
        callback->callOnEpochBegin();
        ctx_compute_n_gradient.run();                    // forward propagation
        ctx_compute_n_gradient.graph_compute_gradeint(); // back propagation
        callback->callOnEpochEnd();
      }
    }
  } else {
    LOG(ERROR) << "Fatal! # no of input is mismatching with the no of graph "
                  "created during construction.\n";
  }
}

void Model::initilizeInputsForTraining(
    std::vector<Tensor<std::float64_t> *> incoming_training_inputs) {
  for (Tensor<std::float64_t> *input : incoming_training_inputs) {
    std::vector<unsigned> dims;
    for (unsigned i = 0; i < input->getNoOfDimensions() - 1; i++)
      dims.push_back(input->getDimensions()[i]);

    dims.push_back(this->batch_size);
    Tensor<std::float64_t> *temp_input =
        new Tensor<std::float64_t>(dims.size(), dims.data(), tf_float64);
    this->local_training_inputs.push_back(temp_input);

    dims.clear();
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
          this_layer->forward(this->layer_training_input_mappings[this_layer],
                              this->batch_size);

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

void Model::getTrainingTensorsForInputLayer() {
  for (Layer *layer : this->input_layers) {
    int i = 0;
    for (Tensor<std::float64_t> *input_tensor :
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