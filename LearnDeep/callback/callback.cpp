// C++ Headers
#include <algorithm>
#include <ranges>

// Library Headers
#include "callback.hpp"
#include <layers/layer_graph.hpp>
#include <layers/layers.hpp>

Callback::Callback(unsigned callback_level) {
  this->callback_level = callback_level;
}

void Callback::onEpochBeginGetTrainableParameter(
    Layer *layer, Layer_Parameter traiable_parameter_no, bool print) {
  if (std::ranges::contains(global_layer_graph.getAllLayers(), layer)) {
    this->layers_on_epoch_begin[layer].push_back(
        {traiable_parameter_no, print});
  }
}

void Callback::onEpochEndGetTrainableParameter(
    Layer *layer, Layer_Parameter traiable_parameter_no, bool print) {
  if (std::ranges::contains(global_layer_graph.getAllLayers(), layer)) {
    this->layers_on_epoch_end[layer].push_back({traiable_parameter_no, print});
  }
}

std::vector<std::vector<tf::tensor>>
Callback::getTrainableParameterEpochOnBegin(
    Layer *layer, Layer_Parameter trainable_parameter_no) {

  std::vector<std::vector<tf::tensor>> layer_parameter_output;

  std::vector<Layer *> layers(
      std::views::keys(this->layer_parameter_on_epoch_begin).begin(),
      std::views::keys(this->layer_parameter_on_epoch_begin).end());

  if (std::ranges::contains(layers, layer)) {
    std::vector<Layer_Parameter> layer_parameter(
        std::views::keys(this->layer_parameter_on_epoch_begin[layer]).begin(),
        std::views::keys(this->layer_parameter_on_epoch_begin[layer]).end());
    if (std::ranges::contains(layer_parameter, trainable_parameter_no))
      layer_parameter_output =
          this->layer_parameter_on_epoch_begin[layer][trainable_parameter_no];
    else
      LOG(ERROR) << "Sever! this parameter was not registered for callback.\n";
  } else {
    LOG(ERROR) << "Sever! Layer was not registered for callback.\n";
  }

  return layer_parameter_output;
}

std::vector<std::vector<tf::tensor>> Callback::getTrainableParameterEpochOnEnd(
    Layer *layer, Layer_Parameter trainable_parameter_no) {

  std::vector<std::vector<tf::tensor>> layer_parameter_output;

  std::vector<Layer *> layers(
      std::views::keys(this->layer_parameter_on_epoch_end).begin(),
      std::views::keys(this->layer_parameter_on_epoch_end).end());

  if (std::ranges::contains(layers, layer)) {
    std::vector<Layer_Parameter> layer_parameter(
        std::views::keys(this->layer_parameter_on_epoch_end[layer]).begin(),
        std::views::keys(this->layer_parameter_on_epoch_end[layer]).end());
    if (std::ranges::contains(layer_parameter, trainable_parameter_no)) {
      layer_parameter_output =
          this->layer_parameter_on_epoch_end[layer][trainable_parameter_no];
    } else
      LOG(ERROR) << "Sever! this parameter was not registered for callback.\n";
  } else {
    LOG(ERROR) << "Sever! Layer was not registered for callback.\n";
  }

  return layer_parameter_output;
}

void Callback::callOnEpochBegin() {
  if (this->print_callback_log)
    std::cout << "Callback before Epoch Begin: \n";
  std::vector<Layer *> layers(std::views::keys(layers_on_epoch_begin).begin(),
                              std::views::keys(layers_on_epoch_begin).end());
  for (Layer *layer : layers) {
    for (auto [layer_parameter, flag] : layers_on_epoch_begin[layer]) {
      std::vector<tf::tensor> epoch_begin_tensors;
      for (tf::tensor tensor :
           layer->getLayerParameter(layer_parameter, flag)) {
        tf::tensor temp_tensor = tensor;

        epoch_begin_tensors.push_back(temp_tensor);
      }
      layer_parameter_on_epoch_begin[layer][layer_parameter].push_back(
          layer->getLayerParameter(layer_parameter, flag));
    }
  }
}

void Callback::callOnEpochEnd() {
  if (this->print_callback_log)
    std::cout << "Callback after Epoch End: \n";
  std::vector<Layer *> layers(std::views::keys(layers_on_epoch_end).begin(),
                              std::views::keys(layers_on_epoch_end).end());

  for (Layer *layer : layers) {
    for (auto [layer_parameter, flag] : layers_on_epoch_end[layer]) {
      std::vector<tf::tensor> epoch_end_tensors;
      for (tf::tensor tensor :
           layer->getLayerParameter(layer_parameter, flag)) {
        tf::tensor temp_tensor = tensor;
        epoch_end_tensors.push_back(temp_tensor);
      }
      this->layer_parameter_on_epoch_end[layer][layer_parameter].push_back(
          epoch_end_tensors);
    }
  }
}
