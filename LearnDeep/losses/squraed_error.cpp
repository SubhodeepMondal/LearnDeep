// Library Header
#include "loss.hpp"

SquaredError::~SquaredError() {
  for (tf::tensor *loss : losses)
    delete loss;
}

void SquaredError::forward() {
  for (unsigned i = 0; i < this->losses.size(); i++) {
    *(this->differences[i]) =
        this->output_targets[i]->sub(*this->output_predicts[i], false);
    *(this->losses[i]) = this->differences[i]->pow(2, false);
  }
}

void SquaredError::setPredictedOutput(
    const std::vector<tf::tensor *> &output_predicts) {
  this->output_predicts = output_predicts;
}

void SquaredError::setTargetOutput(
    const std::vector<tf::tensor *> &output_targets) {
  unsigned idx = 0;
  if (this->output_predicts.size() == output_targets.size()) {
    for (tf::tensor *output_target : output_targets) {
      bool flag_lvl_1 = true;
      if (output_target->getNoOfDimensions() ==
          this->output_predicts[idx]->getNoOfDimensions()) {
        for (unsigned i = 0; i < output_target->getNoOfDimensions(); i++) {
          if (output_target->getDimensions()[i] ==
              this->output_predicts[idx]->getDimensions()[i]) {
            flag_lvl_1 = false;
            break;
          }
        }
      } else {
        LOG(ERROR) << "Fatal! dimensions missmatch!\n";
      }
      if (flag_lvl_1)
        this->output_targets.push_back(output_target);
      else {
        LOG(ERROR) << "Fatal! For target output index: " << idx
                   << " there is a dimensions missmatch with respective "
                      "prediction tensor.";
        LOG(ERROR) << "Tensor dimensions for prediction tensor is:\n";
        this->output_predicts[idx]->print_dimension();
        LOG(ERROR) << "\nTensor dimensions for target tensor is:";
        output_target->print_dimension();
      }
    }
  } else {
    LOG(ERROR) << "Fatal! Espected No Of target outputs are: "
               << this->output_predicts.size()
               << ", but got:  " << output_targets.size() << "\n";
  }
}

const std::vector<tf::tensor *> &SquaredError::getLoss() {
  return this->losses;
}

std::vector<tf::tensor *>
SquaredError::getLossParameter(LossParameter loss_parameter) {
  std::vector<tf::tensor *> output_param;
  switch (loss_parameter) {
  case (LossParameter::output_predict): {
    output_param = this->output_predicts;
    break;
  }
  case (LossParameter::output_target): {
    output_param = this->output_targets;
    break;
  }
  case (LossParameter::loss): {
    output_param = this->losses;
    break;
  }
  default:
    LOG(ERROR)
        << "Fatal! No parameter found! Please check the paramters for loss!\n";
    break;
  }
  return output_param;
}