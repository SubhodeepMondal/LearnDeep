#include "loss_registry.hpp"

#include <absl/log/log.h>

#include <limits>
#include <utility>

#include "loss.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>

namespace {

// Adapts the public behavior interface to the internal Loss ABI. This is the
// only place where a plugin-created loss crosses that boundary.
class BehaviorAdapter final : public Loss {
public:
  BehaviorAdapter(std::string name, std::unique_ptr<tf::loss::behavior> impl)
      : Loss(), name_(std::move(name)), impl_(std::move(impl)) {}

  std::string_view type() const override { return impl_->type(); }

  void forward(std::vector<tf::tensor *> inputs,
               const unsigned batch_size) override {
    impl_->forward(std::move(inputs), batch_size);
  }

  void backward() override { impl_->backward(); }

  void setTargetOutput(const std::vector<tf::tensor> output_predicts) override {
    impl_->setTargetOutput(output_predicts);
  }

  void setPredictedOutput(
      const std::vector<tf::tensor *> &output_targets) override {
    // Prediction inputs are supplied through forward(); the public behavior
    // contract does not need a separate setter.
    (void)output_targets;
  }

  std::float64_t const getScalerLoss() override {
    return impl_->getScalerLoss();
  }

  std::float64_t const getEpochEndScalerLoss() override {
    return impl_->getEpochEndScalerLoss();
  }

  std::float64_t calculate(std::vector<tf::tensor> &predicted_value,
                           std::vector<tf::tensor> &target_output) override {
    return impl_->calculate(predicted_value, target_output);
  }

  std::vector<tf::tensor *>
  getLossParameter(std::string_view loss_parameter) override {
    return impl_->getParameter(loss_parameter);
  }

  std::unordered_set<std::string> parameterNames() const override {
    return impl_->parameterNames();
  }

  unsigned getNoOfInputs() override { return impl_->getNoOfInputs(); }

  std::string getLossName() override { return name_; }

  tf::loss::behavior &impl() { return *impl_; }

private:
  std::string name_;
  std::unique_ptr<tf::loss::behavior> impl_;
};

} // namespace

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

bool LossRegistry::registerBehavior(const std::string &name,
                                     tf::loss::behavior_factory factory,
                                     tf::loss::behavior_setter setter) {
  return registerLoss(
      name,
      [name, factory](const LossConfig &config) -> std::unique_ptr<Loss> {
        std::unique_ptr<tf::loss::behavior> impl = factory(config);
        if (!impl)
          return nullptr;
        return std::make_unique<BehaviorAdapter>(::lossName(config, name),
                                                 std::move(impl));
      },
      [setter](const LossHandle &loss, const std::string &parameter,
               const tf::LossParameterValue &value) {
        auto *adapter = dynamic_cast<BehaviorAdapter *>(loss.get());
        if (!adapter || !setter)
          return false;
        return setter(adapter->impl(), parameter, value);
      });
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

namespace tf {
namespace loss {

bool registerBehavior(std::string_view name, behavior_factory factory,
                      behavior_setter setter) {
  return LossRegistry::instance().registerBehavior(
      std::string(name), std::move(factory), std::move(setter));
}

tf::tensor gradient_of(tf::tensor &input) {
  Graph *graph = GraphManager::instance().getCurrentGraph();
  if (!graph || !input.getPtr())
    return tf::tensor{};
  Tensor<std::float64_t> *gradient = graph->getGradientTensor(input.getPtr());
  if (!gradient)
    return tf::tensor{};
  return tf::tensor(input.dt_type, gradient);
}

} // namespace loss
} // namespace tf
