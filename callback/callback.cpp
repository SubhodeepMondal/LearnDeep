// C++ Headers
#include <algorithm>
#include <ranges>

// Library Headers
#include "callback.hpp"

Callback::Callback(unsigned callback_level) {
  this->callback_level = callback_level;
}

void Callback::onEpochBeginGetTrainableParameter(
    Layer *layer, Layer_Parameter traiable_parameter_no, bool print) {
  if (std::ranges::contains(global_layer_graph.getAllLayers(), layer)) {
    std::pair<Layer_Parameter, bool> p{traiable_parameter_no, print};
    this->layers_on_epoch_begin[layer].push_back(p);
  }
}

void Callback::onEpochEndGetTrainableParameter(
    Layer *layer, Layer_Parameter traiable_parameter_no, bool print) {
  if (std::ranges::contains(global_layer_graph.getAllLayers(), layer)) {
    std::pair<Layer_Parameter, bool> p{traiable_parameter_no, print};
    this->layers_on_epoch_end[layer].push_back(p);
  }
}

std::vector<std::vector<Tensor<std::float64_t> *>>
Callback::getTrainableParameterEpochOnBegin(
    Layer *layer, Layer_Parameter trainable_parameter_no) {

  std::vector<std::vector<Tensor<std::float64_t> *>> layer_parameter_output;

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

std::vector<std::vector<Tensor<std::float64_t> *>>
Callback::getTrainableParameterEpochOnEnd(
    Layer *layer, Layer_Parameter trainable_parameter_no) {

  std::vector<std::vector<Tensor<std::float64_t> *>> layer_parameter_output;

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
  std::vector<Tensor<std::float64_t> *> epoch_begin_tensors;
  std::vector<Layer *> layers(std::views::keys(layers_on_epoch_begin).begin(),
                              std::views::keys(layers_on_epoch_begin).end());
  for (Layer *layer : layers) {
    for (auto [layer_parameter, flag] : layers_on_epoch_begin[layer]) {
      for (Tensor<std::float64_t> *tensor :
           layer->getLayerParameter(layer_parameter, flag)) {
        Tensor<std::float64_t> *temp_tensor =
            new Tensor<std::float64_t>(*tensor);
        epoch_begin_tensors.push_back(temp_tensor);
      }
      layer_parameter_on_epoch_begin[layer][layer_parameter].push_back(
          layer->getLayerParameter(layer_parameter, flag));
    }
  }
}

void Callback::callOnEpochEnd() {
  std::vector<Tensor<std::float64_t> *> epoch_end_tensors;
  if (this->print_callback_log)
    std::cout << "Callback after Epoch End: \n";
  std::vector<Layer *> layers(std::views::keys(layers_on_epoch_end).begin(),
                              std::views::keys(layers_on_epoch_end).end());

  for (Layer *layer : layers) {
    for (auto [layer_parameter, flag] : layers_on_epoch_end[layer]) {

      for (Tensor<std::float64_t> *tensor :
           layer->getLayerParameter(layer_parameter, flag)) {
        Tensor<std::float64_t> *temp_tensor =
            new Tensor<std::float64_t>(*tensor);
        epoch_end_tensors.push_back(temp_tensor);
      }
      this->layer_parameter_on_epoch_end[layer][layer_parameter].push_back(
          epoch_end_tensors);
    }
  }
}
