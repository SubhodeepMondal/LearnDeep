#ifndef _TENSORFLOW_OPTIMIZER_REGISTRY_
#define _TENSORFLOW_OPTIMIZER_REGISTRY_

// C++ Headers
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Library Headers
#include <api/tensor.h>

class Optimizer;

// The only component that knows the internal Optimizer ABI. API handles, Model,
// and Layer call sites route through this facade instead of holding raw
// Optimizer pointers.
class OptimizerRegistry {
public:
  using OptimizerHandle = std::shared_ptr<Optimizer>;
  using Factory =
      std::function<std::unique_ptr<Optimizer>(const OptimizerConfig &)>;
  using ParameterSetter = std::function<bool(
      const OptimizerHandle &, const std::string &,
      const tf::OptimizerParameterValue &)>;

  static OptimizerRegistry &instance();

  bool registerOptimizer(const std::string &name, Factory factory,
                         ParameterSetter setter = {});

  OptimizerHandle create(const std::string &name,
                         const OptimizerConfig &config = {});

  // Track an optimizer created outside a factory, giving it a live id that can
  // be locked and pruned like any registry optimizer.
  OptimizerHandle adopt(std::unique_ptr<Optimizer> optimizer);

  bool setParameter(const std::string &name, const OptimizerHandle &optimizer,
                    const std::string &parameter,
                    const tf::OptimizerParameterValue &value) const;

  bool contains(const std::string &name) const;

  const std::unordered_set<std::string> &names() const;

  // Live-instance tracking: ids distinguish dead objects from foreign ones.
  OptimizerHandle lock(tf::optimizer::optimizer_id id) const;

  bool contains(tf::optimizer::optimizer_id id) const;

  void forget(tf::optimizer::optimizer_id id);

  tf::optimizer::optimizer_id idOf(const OptimizerHandle &optimizer) const;

  std::string_view typeOf(const OptimizerHandle &optimizer) const;

  void createParameterUpdateGraph(const OptimizerHandle &optimizer,
                                  tf::tensor param, tf::tensor &update_param,
                                  tf::tensor gradient_param) const;

  void execute(const OptimizerHandle &optimizer) const;

  std::vector<tf::tensor>
  optimizerParameters(const OptimizerHandle &optimizer,
                      std::string_view parameter) const;

  std::unordered_set<std::string>
  optimizerParameterNames(const OptimizerHandle &optimizer) const;

private:
  OptimizerRegistry() = default;

  std::unordered_set<std::string> optimizer_names_;
  std::unordered_map<std::string, Factory> factories_;
  std::unordered_map<std::string, ParameterSetter> parameter_setters_;
  mutable std::unordered_map<tf::optimizer::optimizer_id,
                             std::weak_ptr<Optimizer>>
      live_optimizers_;
};

inline double optimizerLearningRate(const OptimizerConfig &config,
                                    double fallback = 0.01) {
  const double *learning_rate = config.getAs<double>(
      std::string(tf::optimizer::parameter::learning_rate));
  return learning_rate && *learning_rate > 0.0 ? *learning_rate : fallback;
}

#define LD_OPTIMIZER_CONCAT_INNER(a, b) a##b
#define LD_OPTIMIZER_CONCAT(a, b) LD_OPTIMIZER_CONCAT_INNER(a, b)

#define REGISTER_INTERNAL_OPTIMIZER(name, ...)                                \
  namespace {                                                                  \
  [[maybe_unused]] const bool LD_OPTIMIZER_CONCAT(optimizer_registration_,     \
                                                  __LINE__) = [] {              \
    const std::string optimizer_name = (name);                                 \
    if (!OptimizerRegistry::instance().registerOptimizer(optimizer_name,       \
                                                        __VA_ARGS__))          \
      throw std::runtime_error("Failed to register optimizer: " +             \
                               optimizer_name);                               \
    return true;                                                               \
  }();                                                                         \
  } // namespace

#endif // _TENSORFLOW_OPTIMIZER_REGISTRY_
