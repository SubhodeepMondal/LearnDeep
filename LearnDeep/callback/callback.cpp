// C++ Headers
#include <algorithm>
#include <ranges>

// Library Headers
#include "callback.hpp"
#include <absl/log/log.h>
#include <layers/layer_graph.hpp>
#include <layers/layers.hpp>
#include <losses/loss.hpp>
#include <vector>

Callback::Callback(unsigned callback_level) {
  this->callback_level = callback_level;
}

Callback::~Callback() {
  for (auto tensor : tensor_parameters_on_epoch_begin_vector)
    delete tensor;
  for (auto tensor : tensor_parameters_on_epoch_end_vector)
    delete tensor;

  std::vector<Loss *> loss_ptrs(
      std::views::keys(this->tensor_loss_print_option).begin(),
      std::views::keys(this->tensor_loss_print_option).end());

  for (Loss *loss_ptr : loss_ptrs)
    for (std::vector<tf::tensor *> tensor_vector : this->tensor_loss[loss_ptr])
      for (tf::tensor *tensor_ptr : tensor_vector)
        delete tensor_ptr;
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

void Callback::recordScalerLoss(Loss *const loss_ptr, bool printFlag) {
  this->scaler_loss_print_option[loss_ptr] = printFlag;
}

void Callback::recordTensorLoss(Loss *const loss_ptr, bool printFlag) {
  this->tensor_loss_print_option[loss_ptr] = printFlag;
}

std::vector<std::vector<tf::tensor *>>
Callback::getTrainableParameterEpochOnBegin(
    Layer *layer, Layer_Parameter trainable_parameter_no) {

  std::vector<std::vector<tf::tensor *>> layer_parameter_output;

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

std::vector<std::vector<tf::tensor *>>
Callback::getTrainableParameterEpochOnEnd(
    Layer *layer, Layer_Parameter trainable_parameter_no) {

  std::vector<std::vector<tf::tensor *>> layer_parameter_output;

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

std::vector<std::float64_t> Callback::getScalerLoss(Loss *loss) {
  return this->scalar_loss[loss];
}

std::vector<std::vector<tf::tensor *>> Callback::getTensorLoss(Loss *loss) {
  return this->tensor_loss[loss];
}

void Callback::callOnEpochBegin() {
  if (this->print_callback_log)
    std::cout << "Callback before Epoch Begin: \n";
  std::vector<Layer *> layers(std::views::keys(layers_on_epoch_begin).begin(),
                              std::views::keys(layers_on_epoch_begin).end());
  for (Layer *layer : layers) {
    for (auto [layer_parameter, flag] : layers_on_epoch_begin[layer]) {
      std::vector<tf::tensor *> epoch_begin_tensors;
      for (tf::tensor *tensor :
           layer->getLayerParameter(layer_parameter, flag)) {
        std::vector<unsigned> dims;
        for (unsigned i = 0; i < tensor->getNoOfDimensions(); i++)
          dims.push_back(tensor->getDimensions()[i]);

        this->tensor_parameters_on_epoch_begin_vector.emplace_back(
            new tf::tensor());
        this->tensor_parameters_on_epoch_begin_vector.back()->tf_create(
            dims, tensor->dt_type);
        this->tensor_parameters_on_epoch_begin_vector.back()->tensor_of(
            tensor->getData());
        epoch_begin_tensors.push_back(
            this->tensor_parameters_on_epoch_begin_vector.back());
      }
      layer_parameter_on_epoch_begin[layer][layer_parameter].push_back(
          epoch_begin_tensors);
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
      std::vector<tf::tensor *> epoch_end_tensors;
      for (tf::tensor *tensor :
           layer->getLayerParameter(layer_parameter, flag)) {

        std::vector<unsigned> dims;
        for (unsigned i = 0; i < tensor->getNoOfDimensions(); i++)
          dims.push_back(tensor->getDimensions()[i]);

        this->tensor_parameters_on_epoch_end_vector.emplace_back(
            new tf::tensor());
        this->tensor_parameters_on_epoch_end_vector.back()->tf_create(
            dims, tensor->dt_type);
        this->tensor_parameters_on_epoch_end_vector.back()->tensor_of(
            tensor->getData());
        epoch_end_tensors.push_back(
            this->tensor_parameters_on_epoch_end_vector.back());
      }
      this->layer_parameter_on_epoch_end[layer][layer_parameter].push_back(
          epoch_end_tensors);
    }
  }
}

void Callback::recordLossOnEpochEnd() {
  /* --- scalar loss --- */
  {
    std::vector<Loss *> loss_ptrs(
        std::views::keys(this->scaler_loss_print_option).begin(),
        std::views::keys(this->scaler_loss_print_option).end());

    for (Loss *loss_ptr : loss_ptrs) {
      this->scalar_loss[loss_ptr].push_back(loss_ptr->getScalerLoss());
    }
  }

  /* --- tensor loss --- */
  {
    std::vector<Loss *> loss_ptrs(
        std::views::keys(this->tensor_loss_print_option).begin(),
        std::views::keys(this->tensor_loss_print_option).end());

    for (Loss *loss : loss_ptrs) {
      std::vector<tf::tensor *> temp_tensor;
      for (tf::tensor *tensor : loss->getLossTensor()) {

        std::vector<unsigned> dims;
        for (unsigned i = 0; i < tensor->getNoOfDimensions(); i++)
          dims.push_back(tensor->getDimensions()[i]);

        temp_tensor.emplace_back(new tf::tensor());
        temp_tensor.back()->tf_create(dims, tensor->dt_type);
        temp_tensor.back()->tensor_of(tensor->getData());
      }
      this->tensor_loss[loss].push_back(temp_tensor);
    }
  }
}