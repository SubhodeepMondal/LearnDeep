#include "loss_registry.hpp"

#include <absl/log/log.h>

#include "loss.hpp"

LossRegistry &LossRegistry::instance() {
  static LossRegistry registry;
  return registry;
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
LossRegistry::create(const std::string &name, const LossConfig &config) const {
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

  return LossHandle(created.release());
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
