#ifndef _TENSORFLOW_LAYER_REGISTRY_
#define _TENSORFLOW_LAYER_REGISTRY_

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

class Layer;

struct LayerConfig {
  std::unordered_map<std::string, tf::LayerParameterValue> values;

  bool empty() const { return values.empty(); }

  bool has(const std::string &key) const { return values.count(key) != 0; }

  const tf::LayerParameterValue *get(const std::string &key) const {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
  }

  tf::LayerParameterValue *get(const std::string &key) {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
  }

  void set(std::string key, tf::LayerParameterValue value) {
    values.insert_or_assign(std::move(key), std::move(value));
  }

  template <typename T> const T *getAs(const std::string &key) const {
    const tf::LayerParameterValue *value = get(key);
    return value ? std::get_if<T>(value) : nullptr;
  }

  LayerConfig &with(std::string key, tf::LayerParameterValue value) {
    set(std::move(key), std::move(value));
    return *this;
  }
};

class LayerRegistry {
public:
  using LayerHandle = std::shared_ptr<Layer>;
  using Factory = std::function<std::unique_ptr<Layer>(const LayerConfig &)>;
  using ParameterSetter = std::function<bool(
      const LayerHandle &, const std::string &, const tf::LayerParameterValue &)>;

  static LayerRegistry &instance();

  bool registerLayer(const std::string &name, Factory factory,
                     ParameterSetter setter = {});

  LayerHandle create(const std::string &name,
                     const LayerConfig &config = {}) const;

  bool setParameter(const std::string &name, const LayerHandle &layer,
                    const std::string &parameter,
                    const tf::LayerParameterValue &value) const;

  bool contains(const std::string &name) const;

  const std::unordered_set<std::string> &names() const;

private:
  LayerRegistry() = default;

  std::unordered_set<std::string> layer_names_;
  std::unordered_map<std::string, Factory> factories_;
  std::unordered_map<std::string, ParameterSetter> parameter_setters_;
};

#define LD_CONCAT_INNER(a, b) a##b
#define LD_CONCAT(a, b) LD_CONCAT_INNER(a, b)

#define REGISTER_LAYER(name, ...)                                              \
  namespace {                                                                  \
  [[maybe_unused]] const bool LD_CONCAT(layer_registration_, __LINE__) =        \
      [] {                                                                     \
        const std::string layer_name = (name);                                 \
        if (!LayerRegistry::instance().registerLayer(layer_name, __VA_ARGS__))  \
          throw std::runtime_error("Failed to register layer: " + layer_name); \
        return true;                                                           \
      }();                                                                     \
  } // namespace

#endif // _TENSORFLOW_LAYER_REGISTRY_
