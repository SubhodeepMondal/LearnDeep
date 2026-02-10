#ifndef _TENSORFLOW_OPTIMIZERS_
#define _TENSORFLOW_OPTIMIZERS_

// C++ Headers

// Library Headers
#include <api/tensor.h>

class Optimizer {
protected:
  Graph *optimizer_graph;

public:
  Optimizer();

  ~Optimizer() { delete this->optimizer_graph; }

  virtual void createParameterUpdateGraph(tf::tensor param,
                                          tf::tensor &update_param,
                                          tf::tensor gradient_param) = 0;

  void executeOptimizer();
};

class SGD : public Optimizer {
  double learning_rate;
  std::vector<tf::tensor> scaler_param;

public:
  SGD();

  ~SGD(){};

  void createParameterUpdateGraph(tf::tensor param, tf::tensor &update_param,
                                  tf::tensor gradient_param) override;

  void updateLearningRate(std::float64_t learning_rate);
};

#endif // _TENSORFLOW_OPTIMIZERS_
