#include "loss_registry.hpp"

#include <absl/log/log.h>

#include <limits>

#include "loss.hpp"

LossRegistry &LossRegistry::instance() {
  // Intentionally never destroyed: live-loss deleters may run during static
  // teardown and need a valid registry.
  static LossRegistry *registry = new LossRegistry;
  return *registry;
}

bool LossRegistry::registerLoss(const std::string &name, Factory factory,
                                ParameterSetter setter) {
  if (name.empty() || !factory) {
    LOG(ERROR) << "Cannot register a loss with an empty name or factory.\n";
    return false;
  }

  if (loss_names_.count(name)) {
    LOG(ERROR) << "Loss '" << name << "' is already registered.\n";
    return false;
  }

  loss_names_.insert(name);
  factories_.insert_or_assign(name, std::move(factory));
  if (setter)
    parameter_setters_.insert_or_assign(name, std::move(setter));
  return true;
}

LossRegistry::LossHandle
LossRegistry::create(const std::string &name, const LossConfig &config) {
  auto factory = factories_.find(name);
  if (factory == factories_.end()) {
    LOG(ERROR) << "Unknown loss type '" << name << "'.\n";
    return nullptr;
  }

  std::unique_ptr<Loss> created = factory->second(config);
  if (!created) {
    LOG(ERROR) << "Factory for loss '" << name << "' returned no loss.\n";
    return nullptr;
  }
  return adopt(std::move(created));
}

LossRegistry::LossHandle LossRegistry::adopt(std::unique_ptr<Loss> loss) {
  if (!loss)
    return nullptr;

  const tf::loss::loss_id id = loss->id();
  LossHandle handle(loss.release(), [id](Loss *owned) {
    if (!owned)
      return;
    LossRegistry::instance().forget(id);
    delete owned;
  });
  live_losses_.insert_or_assign(id, handle);
  return handle;
}

bool LossRegistry::setParameter(const std::string &name,
                                const LossHandle &loss,
                                const std::string &parameter,
                                const tf::LossParameterValue &value) const {
  auto setter = parameter_setters_.find(name);
  if (!loss || setter == parameter_setters_.end() ||
      !setter->second(loss, parameter, value)) {
    LOG(ERROR) << "Parameter '" << parameter
               << "' is not settable for loss type '" << name << "'.\n";
    return false;
  }
  return true;
}

bool LossRegistry::contains(const std::string &name) const {
  return loss_names_.count(name) != 0;
}

const std::unordered_set<std::string> &LossRegistry::names() const {
  return loss_names_;
}

LossRegistry::LossHandle LossRegistry::lock(tf::loss::loss_id id) const {
  auto it = live_losses_.find(id);
  return it == live_losses_.end() ? nullptr : it->second.lock();
}

bool LossRegistry::contains(tf::loss::loss_id id) const {
  return static_cast<bool>(lock(id));
}

void LossRegistry::forget(tf::loss::loss_id id) { live_losses_.erase(id); }

tf::loss::loss_id LossRegistry::idOf(const LossHandle &loss) const {
  return loss ? loss->id() : tf::loss::invalid_loss_id;
}

std::string_view LossRegistry::typeOf(const LossHandle &loss) const {
  return loss ? loss->type() : std::string_view{};
}

void LossRegistry::forward(const LossHandle &loss,
                           std::vector<tf::tensor *> inputs,
                           unsigned batch_size) const {
  if (loss)
    loss->forward(inputs, batch_size);
}

void LossRegistry::backward(const LossHandle &loss) const {
  if (loss)
    loss->backward();
}

std::float64_t LossRegistry::calculate(
    const LossHandle &loss, std::vector<tf::tensor> &predicted_output,
    std::vector<tf::tensor> &target_output) const {
  return loss ? loss->calculate(predicted_output, target_output)
              : std::numeric_limits<std::float64_t>::max();
}

std::float64_t LossRegistry::scalerLoss(const LossHandle &loss) const {
  return loss ? loss->getScalerLoss()
              : std::numeric_limits<std::float64_t>::max();
}

std::float64_t LossRegistry::epochEndScalerLoss(const LossHandle &loss) const {
  return loss ? loss->getEpochEndScalerLoss()
              : std::numeric_limits<std::float64_t>::max();
}

std::string LossRegistry::lossName(const LossHandle &loss) const {
  return loss ? loss->getLossName() : std::string{};
}

unsigned LossRegistry::noOfInputs(const LossHandle &loss) const {
  return loss ? loss->getNoOfInputs() : 0;
}

void LossRegistry::setTargetOutput(const LossHandle &loss,
                                   std::vector<tf::tensor> target_output)
    const {
  if (loss)
    loss->setTargetOutput(target_output);
}

std::vector<tf::tensor *>
LossRegistry::lossParameters(const LossHandle &loss,
                             std::string_view parameter) const {
  return loss ? loss->getLossParameter(parameter) : std::vector<tf::tensor *>{};
}

std::unordered_set<std::string>
LossRegistry::lossParameterNames(const LossHandle &loss) const {
  return loss ? loss->parameterNames() : std::unordered_set<std::string>{};
}
