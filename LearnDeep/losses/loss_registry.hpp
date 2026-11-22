#ifndef _TENSORFLOW_LOSS_REGISTRY_
#define _TENSORFLOW_LOSS_REGISTRY_

// C++ Headers
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// Library Headers
#include <api/tensor.h>

class Loss;

struct LossConfig {
  std::unordered_map<std::string, tf::LossParameterValue> values;

  bool empty() const { return values.empty(); }

  bool has(const std::string &key) const { return values.count(key) != 0; }

  const tf::LossParameterValue *get(const std::string &key) const {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
  }

  tf::LossParameterValue *get(const std::string &key) {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
  }

  void set(std::string key, tf::LossParameterValue value) {
    values.insert_or_assign(std::move(key), std::move(value));
  }

  template <typename T> const T *getAs(const std::string &key) const {
    const tf::LossParameterValue *value = get(key);
    return value ? std::get_if<T>(value) : nullptr;
  }

  LossConfig &with(std::string key, tf::LossParameterValue value) {
    set(std::move(key), std::move(value));
    return *this;
  }
};

class LossRegistry {
public:
  using LossHandle = std::shared_ptr<Loss>;
  using Factory = std::function<std::unique_ptr<Loss>(const LossConfig &)>;
  using ParameterSetter = std::function<bool(
      const LossHandle &, const std::string &, const tf::LossParameterValue &)>;

  static LossRegistry &instance();

  bool registerLoss(const std::string &name, Factory factory,
                    ParameterSetter setter = {});

  LossHandle create(const std::string &name,
                    const LossConfig &config = {}) const;

  bool setParameter(const std::string &name, const LossHandle &loss,
                    const std::string &parameter,
                    const tf::LossParameterValue &value) const;

  bool contains(const std::string &name) const;

  const std::unordered_set<std::string> &names() const;

private:
  LossRegistry() = default;

  std::unordered_set<std::string> loss_names_;
  std::unordered_map<std::string, Factory> factories_;
  std::unordered_map<std::string, ParameterSetter> parameter_setters_;
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

#define REGISTER_LOSS(name, ...)                                               \
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
