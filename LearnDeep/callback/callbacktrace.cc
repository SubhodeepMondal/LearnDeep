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

CallbackTrace::CallbackTrace(unsigned callback_level) {
  this->callback_level = callback_level;
}

CallbackTrace::~CallbackTrace() {
  {
    for (auto loss : this->tensor_loss_epoch_end)
      for (auto loss_param : loss.second)
        for (auto tensor_vector_vector : loss_param.second)
          for (auto tensor_vector : tensor_vector_vector)
            delete tensor_vector;
  }
  {
    for (auto loss : this->tensor_loss_batch_end)
      for (auto loss_param : loss.second)
        for (auto tensor_vector_vector_vector : loss_param.second)
          for (auto tensor_vector_vector : tensor_vector_vector_vector)
            for (auto tensor_vector : tensor_vector_vector)
              delete tensor_vector;
  }
}

void CallbackTrace::onEpochBeginGetTrainableParameter(
    Layer *const layer, Layer_Parameter traiable_parameter_no, bool print) {
  if (std::ranges::contains(global_layer_graph.getAllLayers(), layer)) {
    this->layers_on_epoch_begin[layer].push_back(
        {traiable_parameter_no, print});
  }
}

void CallbackTrace::onEpochEndGetTrainableParameter(
    Layer *const layer, Layer_Parameter traiable_parameter_no, bool print) {
  if (std::ranges::contains(global_layer_graph.getAllLayers(), layer)) {
    this->layers_on_epoch_end[layer].push_back({traiable_parameter_no, print});
  }
}

void CallbackTrace::recordTrainableParameterOnBatchBegin(
    Layer *const layer, Layer_Parameter trainable_parameter, bool print) {
  if (std::ranges::contains(global_layer_graph.getAllLayers(), layer)) {
    this->layers_on_batch_begin[layer].push_back({trainable_parameter, print});
  }
}

void CallbackTrace::recordTrainableParameterOnBatchEnd(
    Layer *const layer, Layer_Parameter trainable_parameter, bool print) {
  if (std::ranges::contains(global_layer_graph.getAllLayers(), layer)) {
    this->layers_on_batch_end[layer].push_back({trainable_parameter, print});
  }
}

void CallbackTrace::recordScalerLossEpochEnd(Loss *const loss_ptr) {
  this->losses_on_epoch_end.insert(loss_ptr);
}

void CallbackTrace::recordTensorLossEpochEnd(Loss *const loss_ptr,
                                             Loss_Parameter loss_parameter) {
  this->loss_parameters_on_epoch_end[loss_ptr].push_back(loss_parameter);
}

void CallbackTrace::recordScalerLossBatchEnd(Loss *const loss_ptr) {
  this->losses_on_batch_end.insert(loss_ptr);
}

void CallbackTrace::recordTensorLossBatchEnd(Loss *const loss_ptr,
                                             Loss_Parameter loss_parameter) {
  this->loss_parameters_on_batch_end[loss_ptr].push_back(loss_parameter);
}

std::vector<std::vector<tf::tensor *>>
CallbackTrace::getTrainableParameterEpochOnBegin(
    Layer *const layer, Layer_Parameter trainable_parameter_no) {

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
CallbackTrace::getTrainableParameterEpochOnEnd(
    Layer *const layer, Layer_Parameter trainable_parameter_no) {

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

std::vector<std::vector<std::vector<tf::tensor *>>>
CallbackTrace::getTrainableParameterBatchOnBegin(
    Layer *const layer, Layer_Parameter trainable_parameter_no) {

  std::vector<std::vector<std::vector<tf::tensor *>>> layer_parameter_output;

  std::vector<Layer *> layers(
      std::views::keys(this->layer_parameter_on_batch_begin).begin(),
      std::views::keys(this->layer_parameter_on_batch_begin).end());

  if (std::ranges::contains(layers, layer)) {
    std::vector<Layer_Parameter> layer_parameter(
        std::views::keys(this->layer_parameter_on_batch_begin[layer]).begin(),
        std::views::keys(this->layer_parameter_on_batch_begin[layer]).end());
    if (std::ranges::contains(layer_parameter, trainable_parameter_no))
      layer_parameter_output =
          this->layer_parameter_on_batch_begin[layer][trainable_parameter_no];
    else
      LOG(ERROR) << "Sever! this parameter was not registered for callback.\n";
  } else {
    LOG(ERROR) << "Sever! Layer was not registered for callback.\n";
  }

  return layer_parameter_output;
}

std::vector<std::vector<std::vector<tf::tensor *>>>
CallbackTrace::getTrainableParameterBatchOnEnd(
    Layer *layer, Layer_Parameter trainable_parameter_no) {

  std::vector<std::vector<std::vector<tf::tensor *>>> layer_parameter_output;

  std::vector<Layer *> layers(
      std::views::keys(this->layer_parameter_on_batch_end).begin(),
      std::views::keys(this->layer_parameter_on_batch_end).end());

  if (std::ranges::contains(layers, layer)) {
    std::vector<Layer_Parameter> layer_parameter(
        std::views::keys(this->layer_parameter_on_batch_end[layer]).begin(),
        std::views::keys(this->layer_parameter_on_batch_end[layer]).end());
    if (std::ranges::contains(layer_parameter, trainable_parameter_no)) {
      layer_parameter_output =
          this->layer_parameter_on_batch_end[layer][trainable_parameter_no];
    } else
      LOG(ERROR) << "Sever! this parameter was not registered for callback.\n";
  } else {
    LOG(ERROR) << "Sever! Layer was not registered for callback.\n";
  }

  return layer_parameter_output;
}

std::vector<std::float64_t> CallbackTrace::getScalerLossEpochEnd(Loss *loss) {
  return this->scalar_loss_epoch_end[loss];
}

std::vector<std::vector<tf::tensor *>>
CallbackTrace::getLossParameterEpochEnd(Loss *loss,
                                        Loss_Parameter loss_parameter) {
  return this->tensor_loss_epoch_end[loss][loss_parameter];
}

std::vector<std::vector<std::float64_t>>
CallbackTrace::getScalerLossBatchEnd(Loss *const loss) {
  return this->scalar_loss_batch_end[loss];
}

std::vector<std::vector<std::vector<tf::tensor *>>>
CallbackTrace::getLossParameterBatchEnd(Loss *loss,
                                        Loss_Parameter loss_parameter) {
  return this->tensor_loss_batch_end[loss][loss_parameter];
}

void CallbackTrace::callOnEpochBegin() {
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

        epoch_begin_tensors.emplace_back(new tf::tensor());
        epoch_begin_tensors.back()->tf_create(dims, tensor->dt_type);
        epoch_begin_tensors.back()->tensor_of(tensor->getData());
      }
      layer_parameter_on_epoch_begin[layer][layer_parameter].push_back(
          epoch_begin_tensors);
    }
  }
}

void CallbackTrace::callOnEpochEnd() {

  /* --- layer parameters*/
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

        epoch_end_tensors.emplace_back(new tf::tensor());
        epoch_end_tensors.back()->tf_create(dims, tensor->dt_type);
        epoch_end_tensors.back()->tensor_of(tensor->getData());
      }
      this->layer_parameter_on_epoch_end[layer][layer_parameter].push_back(
          epoch_end_tensors);
    }
  }

  /* --- scalar loss --- */
  {
    for (Loss *loss_ptr : this->losses_on_epoch_end) {
      this->scalar_loss_epoch_end[loss_ptr].push_back(
          loss_ptr->getEpochEndScalerLoss());
    }
  }

  /* --- tensor loss --- */
  {
    for (auto &[loss, loss_parameters] : this->loss_parameters_on_epoch_end) {

      for (Loss_Parameter loss_parameter : loss_parameters) {
        std::vector<tf::tensor *> temp_tensor;
        for (tf::tensor *tensor : loss->getLossParameter(loss_parameter)) {

          std::vector<unsigned> dims;
          for (unsigned i = 0; i < tensor->getNoOfDimensions(); i++)
            dims.push_back(tensor->getDimensions()[i]);

          temp_tensor.emplace_back(new tf::tensor());
          temp_tensor.back()->tf_create(dims, tensor->dt_type);
          temp_tensor.back()->tensor_of(tensor->getData());
        }
        this->tensor_loss_epoch_end[loss][loss_parameter].push_back(
            temp_tensor);
      }
    }
  }
}

void CallbackTrace::callOnBatchBegin(unsigned const epoch_no) {

  if (this->print_callback_log)
    std::cout << "Callback before Epoch Begin: \n";
  std::vector<Layer *> layers(std::views::keys(layers_on_batch_begin).begin(),
                              std::views::keys(layers_on_batch_begin).end());
  for (Layer *layer : layers) {
    for (auto [layer_parameter, flag] : layers_on_batch_begin[layer]) {
      std::vector<tf::tensor *> batch_begin_tensors;
      for (tf::tensor *tensor :
           layer->getLayerParameter(layer_parameter, flag)) {
        std::vector<unsigned> dims;
        for (unsigned i = 0; i < tensor->getNoOfDimensions(); i++)
          dims.push_back(tensor->getDimensions()[i]);

        batch_begin_tensors.emplace_back(new tf::tensor());
        batch_begin_tensors.back()->tf_create(dims, tensor->dt_type);
        batch_begin_tensors.back()->tensor_of(tensor->getData());
      }

      if (layer_parameter_on_batch_begin[layer][layer_parameter].size() ==
          epoch_no)
        layer_parameter_on_batch_begin[layer][layer_parameter].resize(epoch_no +
                                                                      1);

      layer_parameter_on_batch_begin[layer][layer_parameter][epoch_no]
          .push_back(batch_begin_tensors);
    }
  }
}

void CallbackTrace::callOnBatchEnd(unsigned const epoch_no) {
  /* --- layer parameters*/
  if (this->print_callback_log)
    std::cout << "Callback after Epoch End: \n";
  std::vector<Layer *> layers(std::views::keys(layers_on_batch_end).begin(),
                              std::views::keys(layers_on_batch_end).end());

  for (Layer *layer : layers) {
    for (auto [layer_parameter, flag] : layers_on_batch_end[layer]) {
      std::vector<tf::tensor *> batch_end_tensors;
      for (tf::tensor *tensor :
           layer->getLayerParameter(layer_parameter, flag)) {

        std::vector<unsigned> dims;
        for (unsigned i = 0; i < tensor->getNoOfDimensions(); i++)
          dims.push_back(tensor->getDimensions()[i]);

        batch_end_tensors.emplace_back(new tf::tensor());
        batch_end_tensors.back()->tf_create(dims, tensor->dt_type);
        batch_end_tensors.back()->tensor_of(tensor->getData());
      }

      if (layer_parameter_on_batch_end[layer][layer_parameter].size() ==
          epoch_no)
        layer_parameter_on_batch_end[layer][layer_parameter].resize(epoch_no +
                                                                    1);

      layer_parameter_on_batch_end[layer][layer_parameter][epoch_no].push_back(
          batch_end_tensors);
    }
  }

  /* --- scalar loss --- */
  {
    for (Loss *loss_ptr : this->losses_on_batch_end) {
      if (this->scalar_loss_batch_end[loss_ptr].size() == epoch_no)
        this->scalar_loss_batch_end[loss_ptr].resize(epoch_no + 1);

      this->scalar_loss_batch_end[loss_ptr][epoch_no].push_back(
          loss_ptr->getScalerLoss());
    }
  }

  /* --- tensor loss --- */
  {
    for (auto &[loss, loss_parameters] : this->loss_parameters_on_batch_end) {

      for (Loss_Parameter loss_parameter : loss_parameters) {
        std::vector<tf::tensor *> temp_tensor;
        for (tf::tensor *tensor : loss->getLossParameter(loss_parameter)) {

          std::vector<unsigned> dims;
          for (unsigned i = 0; i < tensor->getNoOfDimensions(); i++)
            dims.push_back(tensor->getDimensions()[i]);

          temp_tensor.emplace_back(new tf::tensor());
          temp_tensor.back()->tf_create(dims, tensor->dt_type);
          temp_tensor.back()->tensor_of(tensor->getData());
        }
        if (this->tensor_loss_batch_end[loss][loss_parameter].size() ==
            epoch_no)
          this->tensor_loss_batch_end[loss][loss_parameter].resize(epoch_no +
                                                                   1);

        this->tensor_loss_batch_end[loss][loss_parameter][epoch_no].push_back(
            temp_tensor);
      }
    }
  }
}

bool CallbackTrace::callOnTrainingBegin(tf::model &model) {
  bool flag = this->verifyLayerCallbackSetup(model.get_layers());
  if (flag) {
    this->layers_verification_epoch_begin = true;
    this->layers_verification_epoch_end = true;
    this->layers_verification_batch_begin = true;
    this->layers_verification_batch_end = true;
    flag = flag && this->verifyLossCallbackSetup(model.get_losses());
    if (flag) {
      this->losses_verification_batch_end = true;
      this->losses_verification_epoch_end = true;
    }
  }
  return flag;
}

void CallbackTrace::callOnTrainingEnd() {
  this->layers_verification_epoch_begin = false;
  this->layers_verification_epoch_end = false;
  this->layers_verification_batch_begin = false;
  this->layers_verification_batch_end = false;

  this->losses_verification_batch_end = false;
  this->losses_verification_epoch_end = false;
}

bool CallbackTrace::verifyLayerCallbackSetup(std::vector<Layer *> &layers) {
  bool flag = true;

  // verify layers callback setup on epoch begin
  if (flag) {
    for (Layer *layer : std::views::keys(layers_on_epoch_begin)) {
      auto idx = std::find(layers.begin(), layers.end(), layer);
      if (idx == layers.end()) {
        flag = false;
        std::cerr
            << "Callback set up for layers epoch begin not matching with the "
               "model setup.\n";
        break;
      }
    }
  }

  // verify layers callback setup on epoch end
  if (flag) {
    for (Layer *layer : std::views::keys(layers_on_epoch_end)) {
      auto idx = std::find(layers.begin(), layers.end(), layer);
      if (idx == layers.end()) {
        flag = false;
        std::cerr
            << "Callback set up for layers at epoch end not matching with the "
               "model setup.\n";
        break;
      }
    }
  }

  // verify layers callback setup on batch begin
  if (flag) {
    for (Layer *layer : std::views::keys(layers_on_batch_begin)) {
      auto idx = std::find(layers.begin(), layers.end(), layer);
      if (idx == layers.end()) {
        flag = false;
        std::cerr << "Callback set up for layers at batch begin not matching "
                     "with the "
                     "model setup.\n";
        break;
      }
    }
  }

  // verify layers callback setup on batch end
  if (flag) {
    for (Layer *layer : std::views::keys(layers_on_batch_end)) {
      auto idx = std::find(layers.begin(), layers.end(), layer);
      if (idx == layers.end()) {
        flag = false;
        std::cerr
            << "Callback set up for layers at batch end not matching with the "
               "model setup.\n";
        break;
      }
    }
  }
  return flag;
}

bool CallbackTrace::verifyLossCallbackSetup(std::vector<Loss *> losses) {
  bool flag = true;
  // verify scalar loss callback setup on epoch end
  if (flag) {
    for (Loss *loss : this->losses_on_epoch_end) {
      auto idx = std::find(losses.begin(), losses.end(), loss);
      if (idx == losses.end()) {
        flag = false;
        std::cerr
            << "Callback set up for Loss at epoch end not matching with the "
               "model setup.\n";
        break;
      }
    }
  }

  // verify scalar loss callback setup on batch end
  if (flag) {
    for (Loss *loss : this->losses_on_batch_end) {
      auto idx = std::find(losses.begin(), losses.end(), loss);
      if (idx == losses.end()) {
        flag = false;
        std::cerr
            << "Callback set up for Loss at batch end not matching with the "
               "model setup.\n";
        break;
      }
    }
  }

  // verify tensor loss callback setup on epoch end
  if (flag) {
    for (Loss *loss : std::views::keys(this->loss_parameters_on_epoch_end)) {
      auto idx = std::find(losses.begin(), losses.end(), loss);
      if (idx == losses.end()) {
        flag = false;
        std::cerr
            << "Callback set up for Loss at epoch end not matching with the "
               "model setup.\n";
        break;
      }
    }
  }

  // verify tensor loss callback setup on batch end
  if (flag) {
    for (Loss *loss : std::views::keys(this->loss_parameters_on_batch_end)) {
      auto idx = std::find(losses.begin(), losses.end(), loss);
      if (idx == losses.end()) {
        flag = false;
        std::cerr
            << "Callback set up for Loss at batch end not matching with the "
               "model setup.\n";
        break;
      }
    }
  }
  return flag;
}
