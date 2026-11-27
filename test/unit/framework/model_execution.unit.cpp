// Regression fixtures for Model graph execution.
//
// The probe layers expose multiple inputs/outputs, call counters and captured
// input values so tests can assert exactly which source output was consumed,
// how often a layer ran, and which tensors reached a layer during training.

#include <api/tensor.h>
#include <gtest/gtest.h>
#include <layers/layer_registry.hpp>
#include <layers/layers.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

class Kan34ProbeLayer : public Layer {
  unsigned input_count_;
  unsigned output_count_;
  bool use_second_input_;

  unsigned calculate_calls_ = 0;
  unsigned forward_calls_ = 0;

  tf::tensor calculate_calls_tensor_;
  tf::tensor forward_calls_tensor_;

  std::vector<tf::tensor> outputs_;
  std::vector<tf::tensor *> output_ptrs_;
  std::vector<tf::tensor> training_outputs_;
  std::vector<tf::tensor *> training_output_ptrs_;

  std::vector<tf::tensor> last_calculate_inputs_;
  std::vector<tf::tensor> last_forward_inputs_;

  unsigned source_input(unsigned output_index) const {
    return (use_second_input_ && (output_index % 2 == 1) && input_count_ > 1)
               ? 1u
               : 0u;
  }

  static double factor(unsigned output_index) {
    return 2.0 + static_cast<double>(output_index);
  }

  // Training tensors are built before the first batch is loaded, so their
  // values are still zero when a consumer records them. Give the dual-input
  // probe's outputs distinct shapes instead, so a test can tell which source
  // output reached the consumer.
  tf::tensor makeTrainingOutput(tf::tensor &source, unsigned output_index) {
    if (output_count_ <= 1)
      return source.scale(factor(output_index));

    const unsigned columns = source_input(output_index) == 1 ? 2u : 3u;
    tf::tensor result;
    result.tf_create(std::vector<unsigned>{1u, columns}, tf_float64);
    const unsigned source_elements = source.getPtr()->getNoOfElem();
    const unsigned result_elements = result.getPtr()->getNoOfElem();
    const unsigned copied = std::min(source_elements, result_elements);
    for (unsigned i = 0; i < copied; ++i)
      result.getData()[i] = source.getData()[i] * factor(output_index);
    return result;
  }

  void refreshCounters() {
    calculate_calls_tensor_.tensor_of(static_cast<double>(calculate_calls_),
                                      static_cast<double>(calculate_calls_));
    forward_calls_tensor_.tensor_of(static_cast<double>(forward_calls_),
                                    static_cast<double>(forward_calls_));
  }

  static void captureInputs(const std::vector<tf::tensor> &inputs,
                            std::vector<tf::tensor> &captured) {
    captured.clear();
    captured.reserve(inputs.size());
    for (const tf::tensor &input : inputs)
      if (input.getPtr())
        captured.push_back(input.copy());
  }

  static void captureInputs(const std::vector<const tf::tensor *> &inputs,
                            std::vector<tf::tensor> &captured) {
    captured.clear();
    captured.reserve(inputs.size());
    for (const tf::tensor *input : inputs)
      if (input && input->getPtr())
        captured.push_back(input->copy());
  }

public:
  Kan34ProbeLayer(unsigned input_count, unsigned output_count,
                  bool use_second_input)
      : input_count_(input_count), output_count_(output_count),
        use_second_input_(use_second_input) {
    calculate_calls_tensor_.tf_create(tf_float64, 1u);
    forward_calls_tensor_.tf_create(tf_float64, 1u);
    refreshCounters();
  }

  const std::vector<tf::tensor *> &
  operator()(std::vector<tf::tensor> input_tensors) override {
    ++calculate_calls_;
    refreshCounters();
    captureInputs(input_tensors, last_calculate_inputs_);

    outputs_.clear();
    output_ptrs_.clear();
    outputs_.reserve(output_count_);
    if (input_tensors.size() < input_count_)
      return output_ptrs_;

    for (unsigned i = 0; i < output_count_; ++i) {
      tf::tensor &source = input_tensors[source_input(i)];
      outputs_.emplace_back(source.scale(factor(i), false));
      output_ptrs_.push_back(&outputs_.back());
    }
    return output_ptrs_;
  }

  const std::vector<tf::tensor *> &
  forward(std::vector<const tf::tensor *> &input, unsigned batch_size) override {
    (void)batch_size;
    this->recordTrainingInputs(input);
    ++forward_calls_;
    refreshCounters();
    captureInputs(input, last_forward_inputs_);

    training_outputs_.clear();
    training_output_ptrs_.clear();
    training_outputs_.reserve(output_count_);
    if (input.size() < input_count_ ||
        this->training_input_aliases.size() < input_count_)
      return training_output_ptrs_;

    for (unsigned i = 0; i < output_count_; ++i) {
      tf::tensor &source =
          this->training_input_aliases[source_input(i)];
      training_outputs_.emplace_back(makeTrainingOutput(source, i));
      training_output_ptrs_.push_back(&training_outputs_.back());
    }
    return training_output_ptrs_;
  }

  std::vector<tf::tensor> calculate(std::vector<tf::tensor> inputs) override {
    ++calculate_calls_;
    refreshCounters();
    captureInputs(inputs, last_calculate_inputs_);

    std::vector<tf::tensor> result;
    result.reserve(output_count_);
    if (inputs.size() < input_count_)
      return result;

    for (unsigned i = 0; i < output_count_; ++i)
      result.emplace_back(
          inputs[source_input(i)].scale(factor(i), false));
    return result;
  }

  void backward(const tf::optimizer::optimizer &) override {}

  std::string_view type() const override {
    return this->use_second_input_ ? "kan34_probe_dual_input"
                                   : "kan34_probe_multi_output";
  }

  const std::vector<tf::tensor *> &getOutputTensors() override {
    return this->output_ptrs_;
  }

  std::vector<const tf::tensor *> getInputTrainingTensors() override {
    return this->getTrainingInputAliases();
  }

  const std::vector<tf::tensor *> &getOutputTrainingTensors() override {
    return this->training_output_ptrs_;
  }

  std::vector<tf::tensor *>
  getLayerParameter(std::string_view layer_parameter, bool) override {
    if (layer_parameter == "calculate_calls")
      return {&this->calculate_calls_tensor_};
    if (layer_parameter == "forward_calls")
      return {&this->forward_calls_tensor_};

    if (layer_parameter == "calc_input_0" && !last_calculate_inputs_.empty())
      return {&last_calculate_inputs_[0]};
    if (layer_parameter == "calc_input_1" && last_calculate_inputs_.size() > 1)
      return {&last_calculate_inputs_[1]};
    if (layer_parameter == "fwd_input_0" && !last_forward_inputs_.empty())
      return {&last_forward_inputs_[0]};
    if (layer_parameter == "fwd_input_1" && last_forward_inputs_.size() > 1)
      return {&last_forward_inputs_[1]};
    return {};
  }

  std::unordered_set<std::string> parameterNames() const override {
    return {"calculate_calls", "forward_calls", "calc_input_0",
            "calc_input_1",    "fwd_input_0",   "fwd_input_1"};
  }

  void initializeParameters() override {}
  void resetGradientRecording() override {}
};

REGISTER_LAYER(
    "kan34_probe_multi_output",
    [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      unsigned outputs = 3;
      if (const unsigned *configured = config.getAs<unsigned>("outputs"))
        outputs = *configured;
      return std::make_unique<Kan34ProbeLayer>(1, outputs, false);
    });

REGISTER_LAYER(
    "kan34_probe_dual_input",
    [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      unsigned outputs = 2;
      if (const unsigned *configured = config.getAs<unsigned>("outputs"))
        outputs = *configured;
      return std::make_unique<Kan34ProbeLayer>(2, outputs, true);
    });

tf::tensor vector(double a, double b) {
  tf::tensor result;
  result.tf_create(tf_float64, 1u, 2u);
  result.getData()[0] = a;
  result.getData()[1] = b;
  return result;
}

double firstValue(const tf::tensor &tensor) { return tensor.getData()[0]; }

void expectNear(double actual, double expected) {
  EXPECT_NEAR(actual, expected, 1e-12);
}

tf::layer::layer probe(unsigned inputs, unsigned outputs) {
  const std::string_view type =
      inputs == 1 ? "kan34_probe_multi_output" : "kan34_probe_dual_input";
  return tf::layer::layer(type, LayerConfig{}.with("outputs", outputs));
}

} // namespace

// --- Phase 0 baseline: public handle boundary ---

TEST(Kan34ModelExecution, PublicLayerHandleResolvesLiveLayers) {
  const tf::tensor input = vector(1.0, 1.0);

  tf::layer::dense dense(3);
  ASSERT_TRUE(dense.valid());
  ASSERT_EQ(dense({input}).size(), 1u);

  tf::layer::layer resolved = tf::layer::layer::from_id(dense.id());
  ASSERT_TRUE(resolved.valid());
  EXPECT_EQ(resolved.id(), dense.id());
  EXPECT_EQ(resolved.type(), dense.type());

  const auto direct_outputs = dense.get_output_tensors();
  const auto resolved_outputs = resolved.get_output_tensors();
  ASSERT_EQ(direct_outputs.size(), resolved_outputs.size());
  for (std::size_t i = 0; i < direct_outputs.size(); ++i)
    EXPECT_EQ(direct_outputs[i].getPtr(), resolved_outputs[i].getPtr());
  EXPECT_EQ(tf::layer::layer::producer_of(direct_outputs.front().getPtr()),
            dense.id());
}

TEST(Kan34ModelExecution, ModelOwnsLayerHandlesAfterCallerReleasesThem) {
  const tf::tensor input = vector(1.0, 1.0);

  tf::tensor output;
  {
    tf::layer::dense dense(2);
    output = dense({input}).front();
    dense.initialize_parameters();
    tf::model model({input}, {output});
    EXPECT_EQ(model.get_layers().size(), 1u);
    const auto predicted = model.predict({input});
    ASSERT_EQ(predicted.size(), 1u);
  }

  ASSERT_TRUE(output.getPtr());
}

TEST(Kan34ModelExecution, ProbeLayersExposeDistinctOutputsAndCallCounts) {
  const tf::tensor x = vector(2.0, 2.0);

  tf::layer::layer multi = probe(1, 3);
  ASSERT_TRUE(multi.valid());
  const auto outputs = multi({x});
  ASSERT_EQ(outputs.size(), 3u);
  expectNear(firstValue(outputs[0]), 4.0);
  expectNear(firstValue(outputs[1]), 6.0);
  expectNear(firstValue(outputs[2]), 8.0);

  const auto calls = multi.getParameter("calculate_calls");
  ASSERT_EQ(calls.size(), 1u);
  expectNear(firstValue(calls.front()), 1.0);
}

// --- Phase 2: source-output indexing (enabled after the indexing fix) ---

TEST(Kan34ModelExecution, PredictionUsesCorrectSourceOutput) {
  const tf::tensor x0 = vector(1.0, 1.0);
  const tf::tensor x1 = vector(10.0, 10.0);
  ASSERT_NE(x0.getData(), nullptr);
  ASSERT_NE(x1.getData(), nullptr);

  tf::layer::layer producer = probe(2, 2);
  const auto producer_outputs = producer({x0, x1});
  ASSERT_EQ(producer_outputs.size(), 2u);
  ASSERT_NE(x0.getData(), nullptr);

  tf::layer::layer consumer = probe(1, 1);
  const auto consumer_outputs = consumer({producer_outputs[1]});
  ASSERT_EQ(consumer_outputs.size(), 1u);
  ASSERT_NE(x0.getData(), nullptr);

  tf::model model({x0, x1}, {consumer_outputs[0]});
  ASSERT_NE(x0.getData(), nullptr);
  const auto predicted = model.predict({x0, x1});
  ASSERT_EQ(predicted.size(), 1u);

  // consumer consumes producer output #1 (3*x1), then scales by 2 => 6*x1.
  // The buggy path used output #0 (2*x0) => 4*x0.
  expectNear(firstValue(predicted[0]), 60.0);
}

TEST(Kan34ModelExecution, TrainingUsesCorrectSourceOutput) {
  const tf::tensor x0 = vector(1.0, 1.0);
  const tf::tensor x1 = vector(10.0, 10.0);

  tf::layer::layer producer = probe(2, 2);
  const auto producer_outputs = producer({x0, x1});

  tf::layer::layer consumer = probe(1, 1);
  const auto consumer_outputs = consumer({producer_outputs[1]});

  tf::model model({x0, x1}, {consumer_outputs[0]});
  model.compile(tf::optimizer::type::sgd, tf::loss::type::squared_error);
  model.shuffle(false);

  const tf::tensor target = vector(0.0, 0.0);
  model.fit({x0, x1}, {target}, {}, 1, 2);

  const auto captured = consumer.getParameter("fwd_input_0");
  ASSERT_EQ(captured.size(), 1u);
  tf::tensor captured_input = captured.front();
  // Training must receive producer output #1, whose shape is [1, 2]. The old
  // indexing bug routed output #0 ([1, 3]) into the consumer instead.
  ASSERT_EQ(captured_input.getNoOfDimensions(), 2u);
  EXPECT_EQ(captured_input.getDimensions()[1], 2u);
}

// --- Phase 3: bookkeeping, readiness and diagnostics ---

TEST(Kan34ModelExecution, InputLayersAreDeduplicated) {
  const tf::tensor x0 = vector(1.0, 1.0);
  const tf::tensor x1 = vector(10.0, 10.0);

  tf::layer::layer producer = probe(2, 1);
  const auto outputs = producer({x0, x1});
  ASSERT_EQ(outputs.size(), 1u);

  tf::model model({x0, x1}, {outputs[0]});
  const auto before_params = producer.getParameter("calculate_calls");
  ASSERT_EQ(before_params.size(), 1u);
  const tf::tensor before = before_params.front().copy();

  model.predict({x0, x1});

  const auto after_params = producer.getParameter("calculate_calls");
  ASSERT_EQ(after_params.size(), 1u);
  const tf::tensor after = after_params.front().copy();
  // Both inputs feed the same producer; it must still execute exactly once.
  expectNear(firstValue(after) - firstValue(before), 1.0);
}

TEST(Kan34ModelExecution, PredictionWaitsForAllInputs) {
  const tf::tensor x0 = vector(1.0, 1.0);
  const tf::tensor x1 = vector(10.0, 10.0);

  tf::layer::layer upstream = probe(1, 1);
  const auto upstream_outputs = upstream({x1});

  // This layer consumes a model input (x0) and an upstream output. It is in
  // the initial queue, so it must not run until the upstream producer fills
  // its second slot.
  tf::layer::layer downstream = probe(2, 2);
  const auto downstream_outputs = downstream({x0, upstream_outputs[0]});
  ASSERT_EQ(downstream_outputs.size(), 2u);

  tf::model model({x0, x1}, {downstream_outputs[1]});
  const auto predicted = model.predict({x0, x1});
  ASSERT_EQ(predicted.size(), 1u);
  expectNear(firstValue(predicted[0]), 60.0);
}

// --- Phase 4: fail-fast scheduler ---

TEST(Kan34ModelExecution, UnresolvedDependencyThrows) {
  const tf::tensor x = vector(1.0, 1.0);

  tf::layer::layer producer = probe(1, 1);
  const auto producer_outputs = producer({x});
  tf::layer::layer consumer = probe(1, 1);
  const auto consumer_outputs = consumer({producer_outputs[0]});

  // Dropping the producer leaves the consumer's input without a live producer.
  // Model must reject the graph immediately instead of retrying forever.
  producer = tf::layer::layer{};
  EXPECT_THROW(tf::model model({x}, {consumer_outputs[0]}), std::runtime_error);
}

// --- Phase 6: alias ownership, duplicate slots and construction validation ---

TEST(Kan34ModelExecution, ModelOutputAliasesDoNotStealLayerTensorOwnership) {
  const tf::tensor x = vector(1.0, 1.0);
  tf::layer::layer producer = probe(1, 8);
  const auto outputs = producer({x});
  ASSERT_EQ(outputs.size(), 8u);

  {
    // Eight outputs force several growth steps in Model's output mapping.
    // Without an upfront reserve, vector moves would make those aliases own
    // the layer's tensors and delete them when the model is destroyed.
    tf::model model({x}, {outputs.back()});
    const auto predicted = model.predict({x});
    ASSERT_EQ(predicted.size(), 1u);
  }

  ASSERT_NE(outputs.front().getPtr(), nullptr);
  expectNear(firstValue(outputs.front()), 2.0);
  const auto refreshed = producer({x});
  ASSERT_EQ(refreshed.size(), 8u);
  expectNear(firstValue(refreshed.front()), 2.0);
}

TEST(Kan34ModelExecution, DuplicateSlotsFromOneOutputAreAllRouted) {
  const tf::tensor x = vector(1.0, 1.0);

  tf::layer::layer producer = probe(1, 1);
  const auto producer_outputs = producer({x});
  ASSERT_EQ(producer_outputs.size(), 1u);

  // Both consumer slots consume the same producer output, and the same model
  // output is declared twice. Every matching slot must be filled.
  tf::layer::layer consumer = probe(2, 1);
  const auto consumer_outputs =
      consumer({producer_outputs[0], producer_outputs[0]});
  ASSERT_EQ(consumer_outputs.size(), 1u);

  tf::model model({x}, {consumer_outputs[0], consumer_outputs[0]});
  const auto predicted = model.predict({x});
  ASSERT_EQ(predicted.size(), 2u);
  expectNear(firstValue(predicted[0]), 4.0);
  expectNear(firstValue(predicted[1]), 4.0);
}

TEST(Kan34ModelExecution, UnrelatedDeclaredOutputThrows) {
  const tf::tensor x = vector(1.0, 1.0);
  tf::tensor unrelated;
  unrelated.tf_create(tf_float64, 1u, 2u);

  EXPECT_THROW(tf::model model({x}, {unrelated}), std::runtime_error);
}

TEST(Kan34ModelExecution, DeclaredInputCanBeDeclaredOutput) {
  const tf::tensor x = vector(3.0, 4.0);

  tf::model model({x}, {x});
  const auto predicted = model.predict({x});
  ASSERT_EQ(predicted.size(), 1u);
  expectNear(firstValue(predicted[0]), 3.0);
}
