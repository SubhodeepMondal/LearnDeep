#ifndef _TENSORFLOW_CALLBACK_
#define _TENSORFLOW_CALLBACK_

// C++ Headers
#include <unordered_map>
#include <utility>
#include <vector>

// Library Headers
#include <api/tensor.h>
#include <layers/layer_enum.hpp>
#include <losses/loss_enum.hpp>

class Layer;
class Loss;

class Callback {
public:
  virtual void callOnTrainingBegin() {}

  virtual void callOnEpochBegin() {}

  virtual void callOnBatchBegin() {}

  virtual void callOnBatchEnd() {}

  virtual void callOnEpochEnd() {}

  virtual void callOnTrainingEnd() {}

  virtual bool stopEpoch() { return false; }
};

class CallbackTrace : public Callback {

  unsigned callback_level;
  bool print_callback_log;

  std::vector<tf::tensor *> tensor_parameters_on_epoch_begin_vector;
  std::vector<tf::tensor *> tensor_parameters_on_epoch_end_vector;
  std::unordered_map<Loss *, bool> scaler_loss_print_option;
  std::unordered_map<Loss *, std::vector<std::pair<Loss_Parameter, bool>>>
      tensor_loss_print_option;
  std::unordered_map<Loss *, std::vector<std::float64_t>> scalar_loss;
  std::unordered_map<Loss *,
                     std::unordered_map<Loss_Parameter,
                                        std::vector<std::vector<tf::tensor *>>>>
      tensor_loss;

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
  CallbackTrace(bool print_callback_log)
      : print_callback_log(print_callback_log){};
  CallbackTrace(unsigned callback_level);
  ~CallbackTrace();
  void onEpochBeginGetTrainableParameter(Layer *layer,
                                         Layer_Parameter traiable_parameter_no,
                                         bool print = false);

  void onEpochEndGetTrainableParameter(Layer *layer,
                                       Layer_Parameter trainable_paramter_no,
                                       bool print = false);

  void recordScalerLoss(Loss *loss_ptr, bool printFlag = false);

  void recordTensorLoss(Loss *loss_ptr, Loss_Parameter loss_parameter,
                        bool printFlag = false);

  std::vector<std::vector<tf::tensor *>>
  getTrainableParameterEpochOnBegin(Layer *layer,
                                    Layer_Parameter trainable_parameter_no);

  std::vector<std::vector<tf::tensor *>>
  getTrainableParameterEpochOnEnd(Layer *layer,
                                  Layer_Parameter trainable_parameter_no);

  std::vector<std::float64_t> getScalerLoss(Loss *loss_ptr);

  std::vector<std::vector<tf::tensor *>>
  getLossParameter(Loss *loss_ptr, Loss_Parameter loss_parameter);

  void callOnEpochBegin();

  void callOnEpochEnd();
};

class CallbackHistory : public Callback {};

class CallbackEarlyStopping : public Callback {
  bool early_stopping;
  Loss *loss_ptr;
  int patience;
  bool minimize;
  float min_delta;
  unsigned num_epochs_holding_min_delta;
  std::vector<std::float64_t> scalar_loss;

public:
  CallbackEarlyStopping(Loss *loss, int patience, float min_delta,
                        bool minimize);

  void callOnEpochEnd() override;

  bool stopEpoch() override;
};

#endif // _TENSORFLOW_CALLBACK_