#ifndef _TENSORFLOW_OPTIMIZERS_
#define _TENSORFLOW_OPTIMIZERS_

// C++ Headers
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

// Library Headers
#include <api/tensor.h>

class Optimizer {
private:
  tf::optimizer::optimizer_id id_;

protected:
  Graph *optimizer_graph;

public:
  Optimizer();

  virtual ~Optimizer();

  tf::optimizer::optimizer_id id() const noexcept { return id_; }

  // Canonical registry key. Concrete optimizers override this.
  virtual std::string_view type() const { return {}; }

  virtual void createParameterUpdateGraph(tf::tensor param,
                                          tf::tensor &update_param,
                                          tf::tensor gradient_param) = 0;

  virtual void executeOptimizer();

  virtual std::vector<tf::tensor *>
  getParameter(std::string_view parameter) {
    return {};
  }

  virtual std::unordered_set<std::string> parameterNames() const { return {}; }
};

class SGD : public Optimizer {
  double learning_rate;
  std::vector<tf::tensor> scaler_param;

public:
  SGD();

  explicit SGD(const tf::optimizer::config &config);

  ~SGD() override = default;

  std::string_view type() const override;

  void createParameterUpdateGraph(tf::tensor param, tf::tensor &update_param,
                                  tf::tensor gradient_param) override;

  void updateLearningRate(std::float64_t learning_rate);

  std::float64_t learningRate() const { return learning_rate; }

  std::unordered_set<std::string> parameterNames() const override;
};

#endif // _TENSORFLOW_OPTIMIZERS_
