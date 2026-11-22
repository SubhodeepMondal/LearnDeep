#include "test_scale_layer.hpp"

// Library Headers
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>
#include <layers/layer_registry.hpp>

// C++ Headers
#include <iostream>

TestScaleLayer::TestScaleLayer(double scale) {
  this->scale_tensor_.tf_create(tf_float64, 1u);
  this->setScale(scale);
}

void TestScaleLayer::setScale(double scale) {
  this->scale_ = scale;
  this->scale_tensor_.tensor_of(scale, scale);
}

const std::vector<tf::tensor *> &
TestScaleLayer::operator()(std::vector<tf::tensor> input_tensors) {
  this->recordInputTensors(input_tensors);
  this->layer_outputs.clear();

  if (input_tensors.size() != 1) {
    std::cerr << "TestScaleLayer expects exactly one input tensor.\n";
    return this->layer_outputs;
  }

  std::vector<unsigned> dims;
  for (unsigned i = 0; i < input_tensors[0].getNoOfDimensions(); ++i)
    dims.push_back(input_tensors[0].getDimensions()[i]);

  // graph_flag = false: inference only, no training graph is created here.
  this->output_ = input_tensors[0].scale(this->scale_, false);
  this->layer_outputs.push_back(&this->output_);
  return this->layer_outputs;
}

const std::vector<tf::tensor *> &
TestScaleLayer::forward(std::vector<const tf::tensor *> &input,
                        unsigned batch_size) {
  (void)batch_size;
  this->training_outputs_.clear();
  this->recordTrainingInputs(input);

  if (input.size() != 1 || this->training_input_aliases.empty()) {
    std::cerr << "TestScaleLayer expects exactly one input tensor.\n";
    return this->training_outputs_;
  }

  tf::tensor *training_input = &this->training_input_aliases.front();
  this->training_output_ = training_input->scale(this->scale_);
  this->training_outputs_.push_back(&this->training_output_);
  return this->training_outputs_;
}

std::vector<tf::tensor>
TestScaleLayer::calculate(std::vector<tf::tensor> inputs) {
  std::vector<tf::tensor> result;
  if (inputs.size() != 1)
    return result;

  result.emplace_back(inputs[0].scale(this->scale_, false));
  return result;
}

void TestScaleLayer::backward(Optimizer *) {
  if (this->training_input_aliases.empty())
    return;

  Graph *graph = GraphManager::instance().getCurrentGraph();
  if (!graph)
    return;

  const tf::tensor *training_input = &this->training_input_aliases.front();
  this->grad_input_ =
      tf::tensor(training_input->dt_type,
                 graph->getGradientTensor(training_input->getPtr()));
}

const std::vector<tf::tensor *> &TestScaleLayer::getOutputTensors() {
  return this->layer_outputs;
}

std::vector<const tf::tensor *> TestScaleLayer::getInputTrainingTensors() {
  return this->getTrainingInputAliases();
}

const std::vector<tf::tensor *> &TestScaleLayer::getOutputTrainingTensors() {
  return this->training_outputs_;
}

std::string_view TestScaleLayer::type() const { return "test_scale"; }

std::vector<tf::tensor *>
TestScaleLayer::getLayerParameter(std::string_view layer_parameter, bool) {
  if (layer_parameter == "scale")
    return {&this->scale_tensor_};
  if (layer_parameter == tf::layer::parameter::training_input) {
    std::vector<tf::tensor *> aliases;
    for (const tf::tensor *alias : this->getTrainingInputAliases())
      aliases.push_back(const_cast<tf::tensor *>(alias));
    return aliases;
  }
  if (layer_parameter == tf::layer::parameter::training_output)
    return this->training_outputs_;
  if (layer_parameter == tf::layer::parameter::grad_input)
    return {&this->grad_input_};
  return {};
}

std::unordered_set<std::string> TestScaleLayer::parameterNames() const {
  return {"scale", std::string(tf::layer::parameter::training_input),
          std::string(tf::layer::parameter::training_output),
          std::string(tf::layer::parameter::grad_input)};
}

void TestScaleLayer::initializeParameters() {
  // `scale` is deliberately fixed; nothing to initialize per batch.
}

REGISTER_LAYER(
    "test_scale",
    [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      double scale = 1.0;
      if (const double *configured = config.getAs<double>("scale"))
        scale = *configured;
      else if (const unsigned *configured = config.getAs<unsigned>("scale"))
        scale = static_cast<double>(*configured);
      return std::make_unique<TestScaleLayer>(scale);
    },
    [](const LayerRegistry::LayerHandle &layer, const std::string &name,
       const tf::LayerParameterValue &value) -> bool {
      if (name != "scale")
        return false;

      std::shared_ptr<TestScaleLayer> scale_layer =
          std::dynamic_pointer_cast<TestScaleLayer>(layer);
      if (!scale_layer)
        return false;

      if (const double *scale = std::get_if<double>(&value)) {
        scale_layer->setScale(*scale);
        return true;
      }
      if (const unsigned *scale = std::get_if<unsigned>(&value)) {
        scale_layer->setScale(static_cast<double>(*scale));
        return true;
      }
      return false;
    });

// Referenced by custom/test so the plugin is guaranteed to be loaded and its
// static REGISTER_LAYER initializer has run before any handle is created.
extern "C" void ld_custom_layers_load() {}
