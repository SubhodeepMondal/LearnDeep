// Library Headers
#include "callback.hpp"
#include <absl/log/log.h>
#include <limits>
#include <losses/loss_registry.hpp>

CallbackEarlyStopping::CallbackEarlyStopping(tf::loss::loss_id id, int patience,
                                             float min_delta, bool minimize)
    : loss_id(id), patience(patience), min_delta(min_delta),
      minimize(minimize), early_stopping(false),
      num_epochs_holding_min_delta(0),
      best_loss(minimize ? std::numeric_limits<std::float64_t>::max()
                         : std::numeric_limits<std::float64_t>::lowest()) {}

void CallbackEarlyStopping::callOnEpochEnd() {

  /* --- scalar loss --- */

  LossRegistry::LossHandle loss = LossRegistry::instance().lock(loss_id);
  if (!loss)
    return;

  const std::float64_t current_loss =
      LossRegistry::instance().epochEndScalerLoss(loss);
  this->scalar_loss.push_back(current_loss);

  // The first epoch establishes the baseline; there is nothing to compare yet.
  if (this->scalar_loss.size() == 1) {
    this->best_loss = current_loss;
    return;
  }

  // Standard early-stopping semantics: an epoch only counts as an improvement
  // when the change from the best loss seen so far exceeds min_delta. Stalls
  // (or regressions) accumulate toward `patience`; a real improvement resets
  // the counter and moves the baseline.
  if (this->minimize) {
    if (current_loss < this->best_loss - this->min_delta) {
      this->best_loss = current_loss;
      this->num_epochs_holding_min_delta = 0;
    } else {
      this->num_epochs_holding_min_delta++;
    }
  } else {
    if (current_loss > this->best_loss + this->min_delta) {
      this->best_loss = current_loss;
      this->num_epochs_holding_min_delta = 0;
    } else {
      this->num_epochs_holding_min_delta++;
    }
  }
}

bool CallbackEarlyStopping::stopEpoch() {
  return this->patience > 0 &&
         this->num_epochs_holding_min_delta >=
             static_cast<unsigned>(this->patience);
}
