#include "optimizers.hpp"
#include <core/graph/graph_manager.hpp>

SGD::SGD() : learning_rate(0.01) {}

void SGD::createParameterUpdateGraph(tf::tensor param, tf::tensor &update_param,
                                     tf::tensor gradient_param) {

  GraphManager::instance().pushGraph(this->optimizer_graph);
  scaler_param.push_back(tf::tensor());

  std::vector<unsigned> dims;
  for (unsigned i = 0; i < param.getNoOfDimensions(); i++)
    dims.push_back(param.getDimensions()[i]);
  scaler_param.back().tf_create(dims, param.dt_type);

  scaler_param.back() = gradient_param.scale(learning_rate);
  update_param = param.sub(scaler_param.back());

  GraphManager::instance().popGraph();
}

void SGD::updateLearningRate(std::float64_t learning_rate) {
  this->learning_rate = learning_rate;
}