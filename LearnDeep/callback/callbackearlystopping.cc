// Library Headers
#include "callback.hpp"
#include <absl/log/log.h>
#include <losses/loss.hpp>

CallbackEarlyStopping::CallbackEarlyStopping(Loss *loss, int patience,
                                             float min_delta, bool minimize)
    : loss_ptr(loss), patience(patience), min_delta(min_delta),
      minimize(minimize), early_stopping(false),
      num_epochs_holding_min_delta(0) {}

void CallbackEarlyStopping::callOnEpochEnd() {

  /* --- scalar loss --- */

  this->scalar_loss.push_back(loss_ptr->getScalerLoss());
}

bool CallbackEarlyStopping::stopEpoch() {
  unsigned count = 0;
  if (this->scalar_loss.size()) {
    if (this->minimize) {
      if (this->scalar_loss[this->scalar_loss.size() - 1] <= this->min_delta) {
        this->num_epochs_holding_min_delta++;
      }
      if (this->patience == this->num_epochs_holding_min_delta)
        return true;
      else
        return false;
    } else {
      if (this->scalar_loss[this->scalar_loss.size() - 1] >= this->min_delta) {
        this->num_epochs_holding_min_delta++;
      }
      if (this->patience == this->num_epochs_holding_min_delta)
        return true;
      else
        return false;
    }
  } else
    return false;
}