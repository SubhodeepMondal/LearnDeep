#ifndef _TENSORFLOW_OPTIMIZERS_
#define _TENOSRFLOW_OPTIMIZERS_

// C++ Headers

// Library Headers
#include <api/tensor.h>

class optimizers {
protected:
  double learning_rate;
  void updatePratmeter(tf::tensor *param, tf::tensor gradient_param);
};

class SGD : protected optimizers {
  void updateParameter(tf::tensor *param, tf::tensor *gradient_param) override;
}

#endif // _TENSORFLOW_OPTIMIZERS_
