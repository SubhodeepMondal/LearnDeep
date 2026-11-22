#ifndef _TENSORFLOW_CALLBACK_
#define _TENSORFLOW_CALLBACK_

// C++ Headers
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Library Headers
#include <api/tensor.h>

class Callback {
public:
  virtual bool callOnTrainingBegin(tf::model &model) { return true; }

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
  bool layers_verification_epoch_begin = false;
  bool layers_verification_epoch_end = false;
  bool layers_verification_batch_begin = false;
  bool layers_verification_batch_end = false;
  bool losses_verification_epoch_end = false;
  bool losses_verification_batch_end = false;
  std::unordered_set<tf::loss::loss_id> losses_on_epoch_end;
  std::unordered_map<tf::loss::loss_id, std::vector<std::string>>
      loss_parameters_on_epoch_end;
  std::unordered_set<tf::loss::loss_id> losses_on_batch_end;
  std::unordered_map<tf::loss::loss_id, std::vector<std::string>>
      loss_parameters_on_batch_end;
  std::unordered_map<tf::loss::loss_id, std::vector<std::float64_t>>
      scalar_loss_epoch_end;
  std::unordered_map<
      tf::loss::loss_id,
      std::unordered_map<std::string, std::vector<std::vector<tf::tensor *>>>>
      tensor_loss_epoch_end;
  std::unordered_map<tf::loss::loss_id,
                     std::vector<std::vector<std::float64_t>>>
      scalar_loss_batch_end;
  std::unordered_map<
      tf::loss::loss_id,
      std::unordered_map<std::string,
                         std::vector<std::vector<std::vector<tf::tensor *>>>>>
      tensor_loss_batch_end;

  std::unordered_map<tf::layer::layer_id,
                     std::vector<std::pair<std::string, bool>>>
      layers_on_epoch_begin;
  std::unordered_map<tf::layer::layer_id,
                     std::vector<std::pair<std::string, bool>>>
      layers_on_epoch_end;

  std::unordered_map<tf::layer::layer_id,
                     std::vector<std::pair<std::string, bool>>>
      layers_on_batch_begin;
  std::unordered_map<tf::layer::layer_id,
                     std::vector<std::pair<std::string, bool>>>
      layers_on_batch_end;

  std::unordered_map<
      tf::layer::layer_id,
      std::unordered_map<std::string, std::vector<std::vector<tf::tensor *>>>>
      layer_parameter_on_epoch_begin;
  std::unordered_map<
      tf::layer::layer_id,
      std::unordered_map<std::string, std::vector<std::vector<tf::tensor *>>>>
      layer_parameter_on_epoch_end;
  std::unordered_map<
      tf::layer::layer_id,
      std::unordered_map<std::string,
                         std::vector<std::vector<std::vector<tf::tensor *>>>>>
      layer_parameter_on_batch_begin;
  std::unordered_map<
      tf::layer::layer_id,
      std::unordered_map<std::string,
                         std::vector<std::vector<std::vector<tf::tensor *>>>>>
      layer_parameter_on_batch_end;

  void pruneStaleLayerIds();

  void pruneStaleLossIds();

  bool
  verifyLayerCallbackSetup(const std::vector<tf::layer::layer_id> &layer_ids);

  bool verifyLossCallbackSetup(std::vector<tf::loss::loss_id> losses);

public:
  CallbackTrace(bool print_callback_log)
      : print_callback_log(print_callback_log){};
  CallbackTrace(unsigned callback_level);
  ~CallbackTrace();

  void onEpochBeginGetTrainableParameter(
      const tf::layer::layer &layer_handle,
      const std::string &trainable_parameter, bool print = false);

  void onEpochEndGetTrainableParameter(
      const tf::layer::layer &layer_handle,
      const std::string &trainable_parameter, bool print = false);

  void recordTrainableParameterOnBatchBegin(
      const tf::layer::layer &layer_handle,
      const std::string &trainable_parameter, bool print = false);

  void recordTrainableParameterOnBatchEnd(
      const tf::layer::layer &layer_handle,
      const std::string &trainable_parameter, bool print = false);

  void recordScalerLossEpochEnd(tf::loss::loss_id id);

  void recordTensorLossEpochEnd(tf::loss::loss_id id,
                                std::string_view loss_parameter);

  void recordScalerLossBatchEnd(tf::loss::loss_id id);

  void recordTensorLossBatchEnd(tf::loss::loss_id id,
                                std::string_view loss_parameter);

  /* --- epoch batch vector-tensor --- */
  std::vector<std::vector<tf::tensor *>>
  getTrainableParameterEpochOnBegin(const tf::layer::layer &layer_handle,
                                    const std::string &trainable_parameter);

  std::vector<std::vector<tf::tensor *>>
  getTrainableParameterEpochOnEnd(const tf::layer::layer &layer_handle,
                                  const std::string &trainable_parameter);

  /* --- epoch batch vector-tensor --- */
  std::vector<std::vector<std::vector<tf::tensor *>>>
  getTrainableParameterBatchOnBegin(const tf::layer::layer &layer_handle,
                                    const std::string &trainable_parameter);

  std::vector<std::vector<std::vector<tf::tensor *>>>
  getTrainableParameterBatchOnEnd(const tf::layer::layer &layer_handle,
                                  const std::string &trainable_parameter);

  std::vector<std::float64_t> getScalerLossEpochEnd(tf::loss::loss_id id);

  std::vector<std::vector<tf::tensor *>>
  getLossParameterEpochEnd(tf::loss::loss_id id,
                           std::string_view loss_parameter);

  std::vector<std::vector<std::float64_t>>
  getScalerLossBatchEnd(tf::loss::loss_id id);

  std::vector<std::vector<std::vector<tf::tensor *>>>
  getLossParameterBatchEnd(tf::loss::loss_id id,
                           std::string_view loss_parameter);

  bool callOnTrainingBegin(tf::model &model) override;

  void callOnEpochBegin() override;

  void callOnEpochEnd() override;

  void callOnBatchBegin(unsigned const epoch_no) override;

  void callOnBatchEnd(unsigned const epoch_no) override;

  void callOnTrainingEnd() override;
};

class CallbackEarlyStopping : public Callback {
  bool early_stopping;
  tf::loss::loss_id loss_id;
  int patience;
  bool minimize;
  float min_delta;
  unsigned num_epochs_holding_min_delta;
  std::float64_t best_loss;
  std::vector<std::float64_t> scalar_loss;

public:
  CallbackEarlyStopping(tf::loss::loss_id id, int patience, float min_delta,
                        bool minimize);

  void callOnEpochEnd() override;

  bool stopEpoch() override;
};

#endif // _TENSORFLOW_CALLBACK_
