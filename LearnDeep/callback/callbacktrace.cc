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

namespace {

using LayerRegistrationMap =
    std::unordered_map<tf::layer::layer_id,
                       std::vector<std::pair<std::string, bool>>>;

void eraseDeadLayerIds(LayerRegistrationMap &registrations) {
  for (auto it = registrations.begin(); it != registrations.end();) {
    if (!global_layer_graph.contains(it->first))
      it = registrations.erase(it);
    else
      ++it;
  }
}

} // namespace

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
  {
    for (auto layer : this->layer_parameter_on_epoch_begin)
      for (auto layer_param : layer.second)
        for (auto tensor_vector : layer_param.second)
          for (auto tensor : tensor_vector)
            delete tensor;
    for (auto layer : this->layer_parameter_on_epoch_end)
      for (auto layer_param : layer.second)
        for (auto tensor_vector : layer_param.second)
          for (auto tensor : tensor_vector)
            delete tensor;
    for (auto layer : this->layer_parameter_on_batch_begin)
      for (auto layer_param : layer.second)
        for (auto epoch : layer_param.second)
          for (auto tensor_vector : epoch)
            for (auto tensor : tensor_vector)
              delete tensor;
    for (auto layer : this->layer_parameter_on_batch_end)
      for (auto layer_param : layer.second)
        for (auto epoch : layer_param.second)
          for (auto tensor_vector : epoch)
            for (auto tensor : tensor_vector)
              delete tensor;
  }
}

void CallbackTrace::onEpochBeginGetTrainableParameter(
    const tf::layer::layer &layer_handle,
    const std::string &trainable_parameter, bool print) {
  const tf::layer::layer_id id = layer_handle.id();
  if (layer_handle.valid() && global_layer_graph.contains(id)) {
    this->layers_on_epoch_begin[id].push_back({trainable_parameter, print});
  }
}

void CallbackTrace::onEpochEndGetTrainableParameter(
    const tf::layer::layer &layer_handle,
    const std::string &trainable_parameter, bool print) {
  const tf::layer::layer_id id = layer_handle.id();
  if (layer_handle.valid() && global_layer_graph.contains(id)) {
    this->layers_on_epoch_end[id].push_back({trainable_parameter, print});
  }
}

void CallbackTrace::recordTrainableParameterOnBatchBegin(
    const tf::layer::layer &layer_handle,
    const std::string &trainable_parameter, bool print) {
  const tf::layer::layer_id id = layer_handle.id();
  if (layer_handle.valid() && global_layer_graph.contains(id)) {
    this->layers_on_batch_begin[id].push_back({trainable_parameter, print});
  }
}

void CallbackTrace::recordTrainableParameterOnBatchEnd(
    const tf::layer::layer &layer_handle,
    const std::string &trainable_parameter, bool print) {
  const tf::layer::layer_id id = layer_handle.id();
  if (layer_handle.valid() && global_layer_graph.contains(id)) {
    this->layers_on_batch_end[id].push_back({trainable_parameter, print});
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
    const tf::layer::layer &layer_handle,
    const std::string &trainable_parameter) {

  const tf::layer::layer_id id = layer_handle.id();
  auto layer = this->layer_parameter_on_epoch_begin.find(id);
  if (layer == this->layer_parameter_on_epoch_begin.end()) {
    LOG(ERROR) << "Sever! Layer was not registered for callback.\n";
    return {};
  }

  auto parameter = layer->second.find(trainable_parameter);
  if (parameter == layer->second.end()) {
    LOG(ERROR) << "Sever! this parameter was not registered for callback.\n";
    return {};
  }
  return parameter->second;
}

std::vector<std::vector<tf::tensor *>>
CallbackTrace::getTrainableParameterEpochOnEnd(
    const tf::layer::layer &layer_handle,
    const std::string &trainable_parameter) {

  const tf::layer::layer_id id = layer_handle.id();
  auto layer = this->layer_parameter_on_epoch_end.find(id);
  if (layer == this->layer_parameter_on_epoch_end.end()) {
    LOG(ERROR) << "Sever! Layer was not registered for callback.\n";
    return {};
  }

  auto parameter = layer->second.find(trainable_parameter);
  if (parameter == layer->second.end()) {
    LOG(ERROR) << "Sever! this parameter was not registered for callback.\n";
    return {};
  }
  return parameter->second;
}

std::vector<std::vector<std::vector<tf::tensor *>>>
CallbackTrace::getTrainableParameterBatchOnBegin(
    const tf::layer::layer &layer_handle,
    const std::string &trainable_parameter) {

  const tf::layer::layer_id id = layer_handle.id();
  auto layer = this->layer_parameter_on_batch_begin.find(id);
  if (layer == this->layer_parameter_on_batch_begin.end()) {
    LOG(ERROR) << "Sever! Layer was not registered for callback.\n";
    return {};
  }

  auto parameter = layer->second.find(trainable_parameter);
  if (parameter == layer->second.end()) {
    LOG(ERROR) << "Sever! this parameter was not registered for callback.\n";
    return {};
  }
  return parameter->second;
}

std::vector<std::vector<std::vector<tf::tensor *>>>
CallbackTrace::getTrainableParameterBatchOnEnd(
    const tf::layer::layer &layer_handle,
    const std::string &trainable_parameter) {

  const tf::layer::layer_id id = layer_handle.id();
  auto layer = this->layer_parameter_on_batch_end.find(id);
  if (layer == this->layer_parameter_on_batch_end.end()) {
    LOG(ERROR) << "Sever! Layer was not registered for callback.\n";
    return {};
  }

  auto parameter = layer->second.find(trainable_parameter);
  if (parameter == layer->second.end()) {
    LOG(ERROR) << "Sever! this parameter was not registered for callback.\n";
    return {};
  }
  return parameter->second;
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
  std::vector<tf::layer::layer_id> layer_ids(
      std::views::keys(layers_on_epoch_begin).begin(),
      std::views::keys(layers_on_epoch_begin).end());
  for (tf::layer::layer_id id : layer_ids) {
    std::shared_ptr<Layer> layer = global_layer_graph.lock(id);
    if (!layer)
      continue;

    for (auto [layer_parameter, flag] : layers_on_epoch_begin[id]) {
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
      layer_parameter_on_epoch_begin[id][layer_parameter].push_back(
          epoch_begin_tensors);
    }
  }
}

void CallbackTrace::callOnEpochEnd() {

  /* --- layer parameters*/
  if (this->print_callback_log)
    std::cout << "Callback after Epoch End: \n";
  std::vector<tf::layer::layer_id> layer_ids(
      std::views::keys(layers_on_epoch_end).begin(),
      std::views::keys(layers_on_epoch_end).end());

  for (tf::layer::layer_id id : layer_ids) {
    std::shared_ptr<Layer> layer = global_layer_graph.lock(id);
    if (!layer)
      continue;

    for (auto [layer_parameter, flag] : layers_on_epoch_end[id]) {
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
      this->layer_parameter_on_epoch_end[id][layer_parameter].push_back(
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
  std::vector<tf::layer::layer_id> layer_ids(
      std::views::keys(layers_on_batch_begin).begin(),
      std::views::keys(layers_on_batch_begin).end());
  for (tf::layer::layer_id id : layer_ids) {
    std::shared_ptr<Layer> layer = global_layer_graph.lock(id);
    if (!layer)
      continue;

    for (auto [layer_parameter, flag] : layers_on_batch_begin[id]) {
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

      if (layer_parameter_on_batch_begin[id][layer_parameter].size() ==
          epoch_no)
        layer_parameter_on_batch_begin[id][layer_parameter].resize(epoch_no + 1);

      layer_parameter_on_batch_begin[id][layer_parameter][epoch_no].push_back(
          batch_begin_tensors);
    }
  }
}

void CallbackTrace::callOnBatchEnd(unsigned const epoch_no) {
  /* --- layer parameters*/
  if (this->print_callback_log)
    std::cout << "Callback after Epoch End: \n";
  std::vector<tf::layer::layer_id> layer_ids(
      std::views::keys(layers_on_batch_end).begin(),
      std::views::keys(layers_on_batch_end).end());

  for (tf::layer::layer_id id : layer_ids) {
    std::shared_ptr<Layer> layer = global_layer_graph.lock(id);
    if (!layer)
      continue;

    for (auto [layer_parameter, flag] : layers_on_batch_end[id]) {
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

      if (layer_parameter_on_batch_end[id][layer_parameter].size() == epoch_no)
        layer_parameter_on_batch_end[id][layer_parameter].resize(epoch_no + 1);

      layer_parameter_on_batch_end[id][layer_parameter][epoch_no].push_back(
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

void CallbackTrace::pruneStaleLayerIds() {
  // Phase 3 made layer ownership shared: a handle can die after a callback
  // recorded its id. Drop those registrations so dead layers cannot block
  // training, while still rejecting live layers that belong to another model.
  eraseDeadLayerIds(this->layers_on_epoch_begin);
  eraseDeadLayerIds(this->layers_on_epoch_end);
  eraseDeadLayerIds(this->layers_on_batch_begin);
  eraseDeadLayerIds(this->layers_on_batch_end);
}

bool CallbackTrace::callOnTrainingBegin(tf::model &model) {
  this->pruneStaleLayerIds();
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

bool CallbackTrace::verifyLayerCallbackSetup(
    const std::vector<tf::layer::layer_id> &layer_ids) {
  bool flag = true;

  // verify layers callback setup on epoch begin
  if (flag) {
    for (tf::layer::layer_id id : std::views::keys(layers_on_epoch_begin)) {
      auto idx = std::find(layer_ids.begin(), layer_ids.end(), id);
      if (idx == layer_ids.end()) {
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
    for (tf::layer::layer_id id : std::views::keys(layers_on_epoch_end)) {
      auto idx = std::find(layer_ids.begin(), layer_ids.end(), id);
      if (idx == layer_ids.end()) {
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
    for (tf::layer::layer_id id : std::views::keys(layers_on_batch_begin)) {
      auto idx = std::find(layer_ids.begin(), layer_ids.end(), id);
      if (idx == layer_ids.end()) {
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
    for (tf::layer::layer_id id : std::views::keys(layers_on_batch_end)) {
      auto idx = std::find(layer_ids.begin(), layer_ids.end(), id);
      if (idx == layer_ids.end()) {
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
