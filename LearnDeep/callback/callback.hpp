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

  virtual void callOnBatchBegin(unsigned const epoch_no) {}

  virtual void callOnBatchEnd(unsigned const epoch_no) {}

  virtual void callOnEpochEnd() {}

  virtual void callOnTrainingEnd() {}

  virtual bool stopEpoch() { return false; }
};

class CallbackTrace : public Callback {

  unsigned callback_level;
  bool print_callback_log;
  std::unordered_map<Loss *, bool> scaler_loss_print_option_epoch_end;
  std::unordered_map<Loss *, std::vector<std::pair<Loss_Parameter, bool>>>
      tensor_loss_print_option_epoch_end;
  std::unordered_map<Loss *, bool> scaler_loss_print_option_batch_end;
  std::unordered_map<Loss *, std::vector<std::pair<Loss_Parameter, bool>>>
      tensor_loss_print_option_batch_end;
  std::unordered_map<Loss *, std::vector<std::float64_t>> scalar_loss_epoch_end;
  std::unordered_map<Loss *,
                     std::unordered_map<Loss_Parameter,
                                        std::vector<std::vector<tf::tensor *>>>>
      tensor_loss_epoch_end;
  std::unordered_map<Loss *, std::vector<std::vector<std::float64_t>>>
      scalar_loss_batch_end;
  std::unordered_map<
      Loss *,
      std::unordered_map<Loss_Parameter,
                         std::vector<std::vector<std::vector<tf::tensor *>>>>>
      tensor_loss_batch_end;

  std::unordered_map<Layer *, std::vector<std::pair<Layer_Parameter, bool>>>
      layers_on_epoch_begin;
  std::unordered_map<Layer *, std::vector<std::pair<Layer_Parameter, bool>>>
      layers_on_epoch_end;

  std::unordered_map<Layer *, std::vector<std::pair<Layer_Parameter, bool>>>
      layers_on_batch_begin;
  std::unordered_map<Layer *, std::vector<std::pair<Layer_Parameter, bool>>>
      layers_on_batch_end;

  std::unordered_map<Layer *,
                     std::unordered_map<Layer_Parameter,
                                        std::vector<std::vector<tf::tensor *>>>>
      layer_parameter_on_epoch_begin;
  std::unordered_map<Layer *,
                     std::unordered_map<Layer_Parameter,
                                        std::vector<std::vector<tf::tensor *>>>>
      layer_parameter_on_epoch_end;
  std::unordered_map<
      Layer *,
      std::unordered_map<Layer_Parameter,
                         std::vector<std::vector<std::vector<tf::tensor *>>>>>
      layer_parameter_on_batch_begin;
  std::unordered_map<
      Layer *,
      std::unordered_map<Layer_Parameter,
                         std::vector<std::vector<std::vector<tf::tensor *>>>>>
      layer_parameter_on_batch_end;

public:
  CallbackTrace(bool print_callback_log)
      : print_callback_log(print_callback_log){};
  CallbackTrace(unsigned callback_level);
  ~CallbackTrace();
  void onEpochBeginGetTrainableParameter(Layer *const layer,
                                         Layer_Parameter trainable_parameter_no,
                                         bool print = false);

  void onEpochEndGetTrainableParameter(Layer *const layer,
                                       Layer_Parameter trainable_paramter_no,
                                       bool print = false);

  void
  recordTrainableParameterOnBatchBegin(Layer *const layer,
                                       Layer_Parameter trainable_parameter_no,
                                       bool print = false);

  void
  recordTrainableParameterOnBatchEnd(Layer *const layer,
                                     Layer_Parameter trainable_parameter_no,
                                     bool print = false);

  void recordScalerLossEpochEnd(Loss *const loss_ptr, bool printFlag = false);

  void recordTensorLossEpochEnd(Loss *const loss_ptr,
                                Loss_Parameter loss_parameter,
                                bool printFlag = false);

  void recordScalerLossBatchEnd(Loss *const loss_ptr, bool printFlag = false);

  void recordTensorLossBatchEnd(Loss *const loss_ptr,
                                Loss_Parameter loss_parameter,
                                bool printFlag = false);

  /* --- epoch batch vector-tensor --- */
  std::vector<std::vector<tf::tensor *>>
  getTrainableParameterEpochOnBegin(Layer *const layer,
                                    Layer_Parameter trainable_parameter_no);

  std::vector<std::vector<tf::tensor *>>
  getTrainableParameterEpochOnEnd(Layer *const layer,
                                  Layer_Parameter trainable_parameter_no);

  /* --- epoch batch vector-tensor --- */
  std::vector<std::vector<std::vector<tf::tensor *>>>
  getTrainableParameterBatchOnBegin(Layer *const layer,
                                    Layer_Parameter trainable_parameter_no);

  std::vector<std::vector<std::vector<tf::tensor *>>>
  getTrainableParameterBatchOnEnd(Layer *const layer,
                                  Layer_Parameter trainable_parameter_no);

  std::vector<std::float64_t> getScalerLossEpochEnd(Loss *loss_ptr);

  std::vector<std::vector<tf::tensor *>>
  getLossParameterEpochEnd(Loss *const loss_ptr, Loss_Parameter loss_parameter);

  std::vector<std::vector<std::float64_t>>
  getScalerLossBatchEnd(Loss *const loss_ptr);

  std::vector<std::vector<std::vector<tf::tensor *>>>
  getLossParameterBatchEnd(Loss *const loss_ptr, Loss_Parameter loss_parameter);

  void callOnEpochBegin();

  void callOnEpochEnd();

  void callOnBatchBegin(unsigned const epoch_no);

  void callOnBatchEnd(unsigned const epoch_no);
};

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