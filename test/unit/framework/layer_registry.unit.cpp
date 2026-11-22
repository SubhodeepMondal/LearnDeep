#include "../LinearAlgebraFixtures.unit.hpp"
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>
#include <layers/layer_registry.hpp>

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
  EXPECT_EQ(dense.identity(), static_cast<const void *>(dense.getLayerPtr()));

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
  EXPECT_EQ(missing.identity(), nullptr);
  EXPECT_TRUE(missing({}).empty());
}
