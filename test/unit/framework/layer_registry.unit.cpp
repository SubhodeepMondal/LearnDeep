#include "../LinearAlgebraFixtures.unit.hpp"
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>
#include <layers/layer_graph.hpp>
#include <layers/layer_registry.hpp>

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
