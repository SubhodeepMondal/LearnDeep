#ifndef _TENSORFLOW_OPTIMIZERS_
#define _TENOSRFLOW_OPTIMIZERS_

// C++ Headers

// Library Headers
#include <api/tensor.h>
class Optimizer {
protected:
  double learning_rate;

public:
  virtual void updateParameter(tf::tensor *param, tf::tensor *gradient_param) {}
};

class SGD : public Optimizer {
public:
  SGD() = default;

  ~SGD(){};
  void updateParameter(tf::tensor *param, tf::tensor *gradient_param) override;
};

#endif // _TENSORFLOW_OPTIMIZERS_
