#ifndef _TENSORFLOW_CALLBACK_
#define _TENSORFLOW_CALLBACK_

// C++ Headers
#include <unordered_map>
#include <utility>
#include <vector>

// Library Headers
#include <api/tensor.h>
#include <layers/layer_enum.hpp>

class Layer;

class Callback {

  unsigned callback_level;
  bool print_callback_log;

  std::vector<tf::tensor *> tensor_parameters_on_epoch_begin_vector;
  std::vector<tf::tensor *> tensor_parameters_on_epoch_end_vector;

  std::unordered_map<Layer *, std::vector<std::pair<Layer_Parameter, bool>>>
      layers_on_epoch_begin;
  std::unordered_map<Layer *, std::vector<std::pair<Layer_Parameter, bool>>>
      layers_on_epoch_end;

  std::unordered_map<Layer *,
                     std::unordered_map<Layer_Parameter,
                                        std::vector<std::vector<tf::tensor *>>>>
      layer_parameter_on_epoch_begin;
  std::unordered_map<Layer *,
                     std::unordered_map<Layer_Parameter,
                                        std::vector<std::vector<tf::tensor *>>>>
      layer_parameter_on_epoch_end;

public:
  Callback(bool print_callback_log) : print_callback_log(print_callback_log){};
  Callback(unsigned callback_level);
  ~Callback();
  void onEpochBeginGetTrainableParameter(Layer *layer,
                                         Layer_Parameter traiable_parameter_no,
                                         bool print = false);

  void onEpochEndGetTrainableParameter(Layer *layer,
                                       Layer_Parameter trainable_paramter_no,
                                       bool print = false);

  std::vector<std::vector<tf::tensor *>>
  getTrainableParameterEpochOnBegin(Layer *layer,
                                    Layer_Parameter trainable_parameter_no);

  std::vector<std::vector<tf::tensor *>>
  getTrainableParameterEpochOnEnd(Layer *layer,
                                  Layer_Parameter trainable_parameter_no);

  void callOnEpochBegin();

  void callOnEpochEnd();
};

#endif // _TENSORFLOW_CALLBACK_