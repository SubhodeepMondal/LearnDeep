#include "optimizer_registry.hpp"

#include <absl/log/log.h>

#include "optimizers.hpp"

OptimizerRegistry &OptimizerRegistry::instance() {
  // Intentionally never destroyed: live-optimizer deleters may run during
  // static teardown and need a valid registry.
  static OptimizerRegistry *registry = new OptimizerRegistry;
  return *registry;
}

bool OptimizerRegistry::registerOptimizer(const std::string &name,
                                          Factory factory,
                                          ParameterSetter setter) {
  if (name.empty() || !factory) {
    LOG(ERROR) << "Cannot register an optimizer with an empty name or factory.";
    return false;
  }

  if (optimizer_names_.count(name)) {
    LOG(ERROR) << "Optimizer '" << name << "' is already registered.";
    return false;
  }

  optimizer_names_.insert(name);
  factories_.insert_or_assign(name, std::move(factory));
  if (setter)
    parameter_setters_.insert_or_assign(name, std::move(setter));
  return true;
}

OptimizerRegistry::OptimizerHandle
OptimizerRegistry::create(const std::string &name,
                          const OptimizerConfig &config) {
  auto factory = factories_.find(name);
  if (factory == factories_.end()) {
    LOG(ERROR) << "Unknown optimizer type '" << name << "'.";
    return nullptr;
  }

  std::unique_ptr<Optimizer> created = factory->second(config);
  if (!created) {
    LOG(ERROR) << "Factory for optimizer '" << name
               << "' returned no optimizer.";
    return nullptr;
  }
  return adopt(std::move(created));
}

OptimizerRegistry::OptimizerHandle
OptimizerRegistry::adopt(std::unique_ptr<Optimizer> optimizer) {
  if (!optimizer)
    return nullptr;

  const tf::optimizer::optimizer_id id = optimizer->id();
  OptimizerHandle handle(optimizer.release(), [id](Optimizer *owned) {
    if (!owned)
      return;
    OptimizerRegistry::instance().forget(id);
    delete owned;
  });
  live_optimizers_.insert_or_assign(id, handle);
  return handle;
}

bool OptimizerRegistry::setParameter(
    const std::string &name, const OptimizerHandle &optimizer,
    const std::string &parameter,
    const tf::OptimizerParameterValue &value) const {
  auto setter = parameter_setters_.find(name);
  if (!optimizer || setter == parameter_setters_.end() ||
      !setter->second(optimizer, parameter, value)) {
    LOG(ERROR) << "Parameter '" << parameter
               << "' is not settable for optimizer type '" << name << "'.";
    return false;
  }
  return true;
}

bool OptimizerRegistry::contains(const std::string &name) const {
  return optimizer_names_.count(name) != 0;
}

const std::unordered_set<std::string> &OptimizerRegistry::names() const {
  return optimizer_names_;
}

OptimizerRegistry::OptimizerHandle
OptimizerRegistry::lock(tf::optimizer::optimizer_id id) const {
  auto it = live_optimizers_.find(id);
  return it == live_optimizers_.end() ? nullptr : it->second.lock();
}

bool OptimizerRegistry::contains(tf::optimizer::optimizer_id id) const {
  return static_cast<bool>(lock(id));
}

void OptimizerRegistry::forget(tf::optimizer::optimizer_id id) {
  live_optimizers_.erase(id);
}

tf::optimizer::optimizer_id
OptimizerRegistry::idOf(const OptimizerHandle &optimizer) const {
  return optimizer ? optimizer->id() : tf::optimizer::invalid_optimizer_id;
}

std::string_view
OptimizerRegistry::typeOf(const OptimizerHandle &optimizer) const {
  return optimizer ? optimizer->type() : std::string_view{};
}

void OptimizerRegistry::createParameterUpdateGraph(
    const OptimizerHandle &optimizer, tf::tensor param,
    tf::tensor &update_param, tf::tensor gradient_param) const {
  if (optimizer)
    optimizer->createParameterUpdateGraph(param, update_param, gradient_param);
}

void OptimizerRegistry::execute(const OptimizerHandle &optimizer) const {
  if (optimizer)
    optimizer->executeOptimizer();
}

std::vector<tf::tensor>
OptimizerRegistry::optimizerParameters(const OptimizerHandle &optimizer,
                                       std::string_view parameter) const {
  std::vector<tf::tensor> copies;
  if (!optimizer)
    return copies;

  for (tf::tensor *value : optimizer->getParameter(parameter))
    if (value)
      copies.push_back(*value);
  return copies;
}

std::unordered_set<std::string> OptimizerRegistry::optimizerParameterNames(
    const OptimizerHandle &optimizer) const {
  return optimizer ? optimizer->parameterNames()
                   : std::unordered_set<std::string>{};
}
