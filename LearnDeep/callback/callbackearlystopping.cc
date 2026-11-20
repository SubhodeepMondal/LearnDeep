// Library Headers
#include "callback.hpp"
#include <absl/log/log.h>
#include <limits>
#include <losses/loss.hpp>

CallbackEarlyStopping::CallbackEarlyStopping(Loss *loss, int patience,
                                             float min_delta, bool minimize)
    : loss_ptr(loss), patience(patience), min_delta(min_delta),
      minimize(minimize), early_stopping(false),
      num_epochs_holding_min_delta(0),
      best_loss(minimize ? std::numeric_limits<std::float64_t>::max()
                         : std::numeric_limits<std::float64_t>::lowest()) {}

void CallbackEarlyStopping::callOnEpochEnd() {

  /* --- scalar loss --- */

  const std::float64_t current_loss = loss_ptr->getEpochEndScalerLoss();
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
