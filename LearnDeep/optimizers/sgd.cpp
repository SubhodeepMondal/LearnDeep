#include "optimizers.hpp"

#include <core/graph/graph_manager.hpp>

#include "optimizer_registry.hpp"

namespace {

// Balance GraphManager's stack even if recording the update graph throws.
class OptimizerGraphScope {
public:
  explicit OptimizerGraphScope(Graph *graph) {
    GraphManager::instance().pushGraph(graph);
  }

  ~OptimizerGraphScope() { GraphManager::instance().popGraph(); }
};

} // namespace

SGD::SGD() : learning_rate(0.01) {}

SGD::SGD(const tf::optimizer::config &config) : SGD() {
  this->learning_rate = optimizerLearningRate(config, this->learning_rate);
}

std::string_view SGD::type() const { return tf::optimizer::type::sgd; }

void SGD::createParameterUpdateGraph(tf::tensor param, tf::tensor &update_param,
                                     tf::tensor gradient_param) {

  OptimizerGraphScope scope(this->optimizer_graph);
  scaler_param.push_back(tf::tensor());

  std::vector<unsigned> dims;
  for (unsigned i = 0; i < param.getNoOfDimensions(); i++)
    dims.push_back(param.getDimensions()[i]);
  scaler_param.back().tf_create(dims, param.dt_type);

  scaler_param.back() = gradient_param.scale(learning_rate);
  update_param = param.sub(scaler_param.back());
}

void SGD::updateLearningRate(std::float64_t learning_rate) {
  this->learning_rate = learning_rate;
}

std::unordered_set<std::string> SGD::parameterNames() const {
  return {std::string(tf::optimizer::parameter::learning_rate)};
}

REGISTER_INTERNAL_OPTIMIZER(
    std::string(tf::optimizer::type::sgd),
    [](const OptimizerConfig &config) -> std::unique_ptr<Optimizer> {
      return std::make_unique<SGD>(config);
    },
    [](const OptimizerRegistry::OptimizerHandle &optimizer,
       const std::string &parameter,
       const tf::OptimizerParameterValue &value) {
      auto *sgd = dynamic_cast<SGD *>(optimizer.get());
      const double *learning_rate = std::get_if<double>(&value);
      if (!sgd || !learning_rate ||
          parameter != tf::optimizer::parameter::learning_rate)
        return false;
      sgd->updateLearningRate(*learning_rate);
      return true;
    });
