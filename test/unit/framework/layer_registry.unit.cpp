#include "../LinearAlgebraFixtures.unit.hpp"
#include <LearnDeep/api/tensor.h>
#include <callback/callback.hpp>
#include <gtest/gtest.h>
#include <layers/dense.hpp>
#include <layers/layer_graph.hpp>
#include <layers/layer_registry.hpp>

#include <algorithm>
#include <memory>
#include <stdfloat>

TEST_F(FrameworkTest, LayerRegistryContainsBuiltInLayers) {
  const std::unordered_set<std::string> &names =
      LayerRegistry::instance().names();

  EXPECT_TRUE(names.count("dense"));
  EXPECT_TRUE(names.count("relu"));
  EXPECT_TRUE(names.count("softmax"));
}

TEST_F(FrameworkTest, GenericLayerHandleSupportsRegistryAndParameters) {
  tf::tensor x;
  tf::tensor weight;
  tf::tensor bias;

  x.tf_create(tf_float64, 3, 2);
  weight.tf_create(tf_float64, 4, 3);
  weight.tensor_of(1.0, 1.0);
  bias.tf_create(tf_float64, 4, 1);
  bias.tensor_of(0.5, 0.5);

  LayerConfig config;
  config.set("units", 4u);
  config.set("weight", weight);

  tf::layer::layer dense("dense", config);
  ASSERT_EQ(dense.type(), "dense");
  ASSERT_TRUE(dense.valid());
  EXPECT_TRUE(static_cast<bool>(dense));
  EXPECT_NE(dense.id(), tf::layer::invalid_layer_id);

  tf::layer::layer dense_copy = dense;
  EXPECT_EQ(dense_copy.id(), dense.id());

  std::unordered_set<std::string> parameter_names = dense.parameterNames();
  EXPECT_TRUE(parameter_names.count("weight"));
  EXPECT_TRUE(parameter_names.count("bias"));
  EXPECT_TRUE(parameter_names.count("training_weight"));

  std::vector<tf::tensor> outputs = dense({x});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].getNoOfDimensions(), 2u);

  dense.setParameter("bias", bias);
  dense.initialize_parameters();

  std::vector<tf::tensor> bias_values = dense.getParameter("bias");
  ASSERT_EQ(bias_values.size(), 1u);
  EXPECT_NEAR(bias_values[0].getData()[0], 0.5, 1e-6);

  EXPECT_EQ(dense.forward({x}, 2).size(), 1u);
  EXPECT_EQ(dense.calculate({x}).size(), 1u);
}

TEST_F(FrameworkTest, GenericLayerHandleRejectsUnknownType) {
  tf::layer::layer missing("not_a_layer");

  EXPECT_TRUE(missing.type().empty());
  EXPECT_FALSE(missing.valid());
  EXPECT_FALSE(static_cast<bool>(missing));
  EXPECT_EQ(missing.id(), tf::layer::invalid_layer_id);
  EXPECT_TRUE(missing({}).empty());
}

TEST_F(FrameworkTest, LayerOwnershipIsSharedByModelAndApiHandle) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  tf::layer::layer_id id = tf::layer::invalid_layer_id;
  std::unique_ptr<tf::model> model;

  {
    tf::layer::dense dense(4);
    id = dense.id();
    ASSERT_NE(id, tf::layer::invalid_layer_id);

    std::vector<tf::tensor> outputs = dense({x});
    model = std::make_unique<tf::model>(std::vector<tf::tensor>{x}, outputs);
    EXPECT_TRUE(global_layer_graph.contains(id));
  }

  EXPECT_TRUE(global_layer_graph.contains(id));
  EXPECT_NE(global_layer_graph.lock(id), nullptr);

  model.reset();
  EXPECT_FALSE(global_layer_graph.contains(id));
}

TEST_F(FrameworkTest, LayerGraphReportsEveryConsumerOfASharedTensor) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  tf::layer::dense dense(4);
  tf::layer::relu relu;
  tf::layer::softmax softmax;

  dense({x});
  relu({x});
  softmax({x});

  const std::vector<tf::layer::layer_id> consumers =
      global_layer_graph.getLayersOfIncomingTensor(x.getPtr());

  EXPECT_NE(std::find(consumers.begin(), consumers.end(), dense.id()),
            consumers.end());
  EXPECT_NE(std::find(consumers.begin(), consumers.end(), relu.id()),
            consumers.end());
  EXPECT_NE(std::find(consumers.begin(), consumers.end(), softmax.id()),
            consumers.end());
}

TEST_F(FrameworkTest, LayerGraphTracksOnlyTheMostRecentInputAlias) {
  tf::tensor old_input;
  old_input.tf_create(tf_float64, 3, 2);
  tf::tensor new_input;
  new_input.tf_create(tf_float64, 3, 2);

  tf::layer::dense dense(4);
  dense({old_input});
  dense({new_input});

  const std::vector<tf::layer::layer_id> old_consumers =
      global_layer_graph.getLayersOfIncomingTensor(old_input.getPtr());
  EXPECT_EQ(std::find(old_consumers.begin(), old_consumers.end(), dense.id()),
            old_consumers.end());

  const std::vector<tf::layer::layer_id> new_consumers =
      global_layer_graph.getLayersOfIncomingTensor(new_input.getPtr());
  EXPECT_NE(std::find(new_consumers.begin(), new_consumers.end(), dense.id()),
            new_consumers.end());
}

TEST_F(FrameworkTest, ModelSchedulesEveryConsumerOfASharedInputTensor) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  tf::layer::dense dense(4);
  tf::layer::relu relu;
  const std::vector<tf::tensor> dense_output = dense({x});
  const std::vector<tf::tensor> relu_output = relu({x});

  tf::model model({x}, {dense_output[0], relu_output[0]});
  const std::vector<tf::layer::layer_id> scheduled = model.get_layers();

  // Both layers consume x; doTensorAndLayerMappings() must schedule both.
  EXPECT_NE(std::find(scheduled.begin(), scheduled.end(), dense.id()),
            scheduled.end());
  EXPECT_NE(std::find(scheduled.begin(), scheduled.end(), relu.id()),
            scheduled.end());
}

TEST_F(FrameworkTest, SetParameterUsesCanonicalLayerTypeForRegistryAliases) {
  ASSERT_TRUE(LayerRegistry::instance().registerLayer(
      "dense_alias_for_set_parameter",
      [](const LayerConfig &config) -> std::unique_ptr<Layer> {
        const unsigned *units = config.getAs<unsigned>("units");
        return std::make_unique<Dense>(units ? *units : 1u);
      }));

  tf::tensor x;
  x.tf_create(tf_float64, 2, 2);
  tf::tensor bias;
  bias.tf_create(tf_float64, 2, 1);
  bias.tensor_of(0.75, 0.75);

  tf::layer::layer alias("dense_alias_for_set_parameter");
  ASSERT_TRUE(alias.valid());
  EXPECT_EQ(alias.type(), "dense");

  alias.setParameter("bias", bias);
  alias({x});
  alias.initialize_parameters();
  const std::vector<tf::tensor> values = alias.getParameter("bias");
  ASSERT_EQ(values.size(), 1u);
  EXPECT_NEAR(values[0].getData()[0], 0.75, 1e-12);
}

TEST_F(FrameworkTest, StaleLayerCallbackIdsDoNotBlockTraining) {
  tf::tensor x;
  x.tf_create(tf_float64, 2, 2);
  x.tensor_of(0.0, 1.0);

  tf::callback::trace trace;
  tf::layer::layer_id stale_id = tf::layer::invalid_layer_id;
  {
    tf::layer::dense dead(3);
    stale_id = dead.id();
    trace.record_parameter_on_batch_end(dead, "weight", false);
  }
  ASSERT_FALSE(global_layer_graph.contains(stale_id));

  tf::layer::dense live(3);
  trace.record_parameter_on_batch_end(live, "weight", false);
  const std::vector<tf::tensor> outputs = live({x});

  tf::model model({x}, outputs);
  model.compile(tf::optimizer::type::sgd, tf::loss::type::squared_error);
  EXPECT_TRUE(trace.callback()->callOnTrainingBegin(model));

  tf::tensor target;
  target.tf_create(tf_float64, 3, 2);
  target.tensor_of(0.0, 0.0);
  model.shuffle(false);
  model.fit({x}, {target}, {trace.callback()}, 1, 2);

  EXPECT_EQ(trace.get_parameter_on_batch_end(live, "weight").size(), 1u);
}

TEST_F(FrameworkTest, TrainingInputStaysOwnedByLayerAfterHandleDies) {
  tf::tensor x;
  x.tf_create(tf_float64, 2, 2);
  x.tensor_of(0.25, 0.25);

  tf::layer::layer_id id = tf::layer::invalid_layer_id;
  std::unique_ptr<tf::model> model;
  {
    tf::layer::dense dense(3);
    id = dense.id();
    const std::vector<tf::tensor> outputs = dense({x});
    ASSERT_EQ(dense.forward({x}, 2).size(), 1u);
    model = std::make_unique<tf::model>(std::vector<tf::tensor>{x}, outputs);
  }

  const std::shared_ptr<Layer> internal = global_layer_graph.lock(id);
  ASSERT_NE(internal, nullptr);
  const std::vector<tf::tensor *> training_input =
      internal->getLayerParameter("training_input");
  ASSERT_EQ(training_input.size(), 1u);
  EXPECT_NEAR(training_input.front()->getData()[0], 0.25, 1e-12);
}
