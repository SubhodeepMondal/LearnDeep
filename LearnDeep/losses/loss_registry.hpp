#ifndef _TENSORFLOW_LOSS_REGISTRY_
#define _TENSORFLOW_LOSS_REGISTRY_

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

class Loss;

// The only component that knows the internal Loss ABI. API handles, Model,
// Callback and history all route through this facade.
class LossRegistry {
public:
  using LossHandle = std::shared_ptr<Loss>;
  using Factory = std::function<std::unique_ptr<Loss>(const LossConfig &)>;
  using ParameterSetter = std::function<bool(
      const LossHandle &, const std::string &, const tf::LossParameterValue &)>;

  static LossRegistry &instance();

  bool registerLoss(const std::string &name, Factory factory,
                    ParameterSetter setter = {});

  // Public behavior API entry point: adapts a plugin to the internal Loss ABI.
  bool registerBehavior(const std::string &name,
                        tf::loss::behavior_factory factory,
                        tf::loss::behavior_setter setter = {});

  LossHandle create(const std::string &name, const LossConfig &config = {});

  // Track a loss created outside a factory (e.g. a test double), giving it a
  // live id that callbacks can lock and prune like any registry loss.
  LossHandle adopt(std::unique_ptr<Loss> loss);

  bool setParameter(const std::string &name, const LossHandle &loss,
                    const std::string &parameter,
                    const tf::LossParameterValue &value) const;

  bool contains(const std::string &name) const;

  const std::unordered_set<std::string> &names() const;

  // Live-instance tracking. Callbacks hold ids only, so they need a way to
  // tell a dead id from a foreign (live) one.
  LossHandle lock(tf::loss::loss_id id) const;

  bool contains(tf::loss::loss_id id) const;

  void forget(tf::loss::loss_id id);

  tf::loss::loss_id idOf(const LossHandle &loss) const;

  std::string_view typeOf(const LossHandle &loss) const;

  void forward(const LossHandle &loss, std::vector<tf::tensor *> inputs,
               unsigned batch_size) const;

  void backward(const LossHandle &loss) const;

  std::float64_t calculate(const LossHandle &loss,
                           std::vector<tf::tensor> &predicted_output,
                           std::vector<tf::tensor> &target_output) const;

  std::float64_t scalerLoss(const LossHandle &loss) const;

  std::float64_t epochEndScalerLoss(const LossHandle &loss) const;

  std::string lossName(const LossHandle &loss) const;

  unsigned noOfInputs(const LossHandle &loss) const;

  void setTargetOutput(const LossHandle &loss,
                       std::vector<tf::tensor> target_output) const;

  std::vector<tf::tensor *> lossParameters(const LossHandle &loss,
                                           std::string_view parameter) const;

  std::unordered_set<std::string> lossParameterNames(
      const LossHandle &loss) const;

private:
  LossRegistry() = default;

  std::unordered_set<std::string> loss_names_;
  std::unordered_map<std::string, Factory> factories_;
  std::unordered_map<std::string, ParameterSetter> parameter_setters_;
  mutable std::unordered_map<tf::loss::loss_id, std::weak_ptr<Loss>>
      live_losses_;
};

inline LossConfig
makeLossConfig(const std::vector<Tensor<std::float64_t> *> &inputs,
               const std::string &name) {
  LossConfig config;
  config.set(std::string(tf::loss::parameter::inputs), inputs);
  config.set(std::string(tf::loss::parameter::name), name);
  return config;
}

inline std::vector<Tensor<std::float64_t> *> lossInputs(const LossConfig &config) {
  const auto *inputs = config.getAs<std::vector<Tensor<std::float64_t> *>>(
      std::string(tf::loss::parameter::inputs));
  return inputs ? *inputs : std::vector<Tensor<std::float64_t> *>{};
}

inline std::string lossName(const LossConfig &config,
                            const std::string &fallback) {
  const std::string *name =
      config.getAs<std::string>(std::string(tf::loss::parameter::name));
  return name ? *name : fallback;
}

#define LD_LOSS_CONCAT_INNER(a, b) a##b
#define LD_LOSS_CONCAT(a, b) LD_LOSS_CONCAT_INNER(a, b)

#define REGISTER_INTERNAL_LOSS(name, ...)                                      \
  namespace {                                                                  \
  [[maybe_unused]] const bool LD_LOSS_CONCAT(loss_registration_, __LINE__) =     \
      [] {                                                                     \
        const std::string loss_name = (name);                                   \
        if (!LossRegistry::instance().registerLoss(loss_name, __VA_ARGS__))      \
          throw std::runtime_error("Failed to register loss: " + loss_name);    \
        return true;                                                           \
      }();                                                                     \
  } // namespace

#endif // _TENSORFLOW_LOSS_REGISTRY_
