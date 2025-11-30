#include <core/Layers/Dense.hh>

Tensor<std::float64_t> *Dense::Dense(unsigned no_of_neuron,
                                     unsigned no_of_feature) {
  this->no_of_neuron = no_of_neuron;
  this->no_of_feature = no_of_feature;

  this->initiaizeInputs();
  this->inttializeOutput();
  return output;
}

Tensor<std::float64_t> *
Dense::Dense(std::vector<Tensor<std::float64_t> *> inputs) {
  if (inputs.size())
    this->inputs.push_back(inputs[0]);
  else
    LOG(ERROR) << "Fatal! No input for the dense layer.\n";

  this->no_of_feature = inputs[0].getNoOfDimensions()[1];
  this->no_of_neuron = inputs[0].getNoOfDimensions()[0];
  this->initializeInputs();
  this->initiaizeOutput();
  return output;
}

void Dense::initiaizeInputs() {
  this->batch_size = 128; // default
  input = new Tensor<std::float64_t>(no_of_fature, batch_size);
  weights = new Tensor<std::float64_t>(batch_size, no_of_neuron);
  bias = new Tensor<std::float64_t>(no_of_neuron, ) inputs =
      std::vector<Tensor<std::float64_t> *>(
          {new Tensor<std::float64_t>(no_of_feature, batch_size),
           new Tensor<std::float64_t>(batch_size, no_of_neuron)});
}

void Dense::initiaizeOutput() {
  output = std::vector<Tensor<std::float64_t> *>(
      {new Tensor<std::float64_t>(2, no_of_feature, batch_size)});
}