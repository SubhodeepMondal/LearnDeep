#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>
#include <optimizers/optimizers.hpp>
#include <optimizers/optimizer_registry.hpp>

#include <string>
#include <unordered_set>

TEST(OptimizerRegistryTest, ContainsBuiltInOptimizers) {
  const std::unordered_set<std::string> &names =
      OptimizerRegistry::instance().names();

  EXPECT_TRUE(names.count(std::string(tf::optimizer::type::sgd)));
}

TEST(OptimizerRegistryTest, DuplicateRegistrationIsRejected) {
  const std::string name = "duplicate_optimizer_regression";
  const OptimizerRegistry::Factory factory =
      [](const OptimizerConfig &) -> std::unique_ptr<Optimizer> {
    return nullptr;
  };

  EXPECT_TRUE(
      OptimizerRegistry::instance().registerOptimizer(name, factory));
  EXPECT_FALSE(
      OptimizerRegistry::instance().registerOptimizer(name, factory));
  EXPECT_TRUE(OptimizerRegistry::instance().contains(name));
}

TEST(OptimizerRegistryTest, GenericHandleSupportsRegistryConstruction) {
  OptimizerConfig config;
  config.set(std::string(tf::optimizer::parameter::learning_rate), 0.05);

  auto sgd = tf::optimizer::optimizer(tf::optimizer::type::sgd, config);
  ASSERT_TRUE(sgd.valid());
  EXPECT_TRUE(static_cast<bool>(sgd));
  EXPECT_EQ(sgd.type(), tf::optimizer::type::sgd);
  EXPECT_NE(sgd.id(), tf::optimizer::invalid_optimizer_id);
  auto locked = OptimizerRegistry::instance().lock(sgd.id());
  ASSERT_NE(locked, nullptr);
  auto *internal_sgd = dynamic_cast<SGD *>(locked.get());
  ASSERT_NE(internal_sgd, nullptr);
  EXPECT_DOUBLE_EQ(internal_sgd->learningRate(), 0.05);

  sgd.setParameter(tf::optimizer::parameter::learning_rate, 0.2);
  EXPECT_DOUBLE_EQ(internal_sgd->learningRate(), 0.2);

  EXPECT_TRUE(OptimizerRegistry::instance().contains(sgd.id()));
  EXPECT_EQ(OptimizerRegistry::instance().lock(sgd.id()), locked);

  auto sgd_copy = sgd;
  EXPECT_EQ(sgd_copy.id(), sgd.id());
}

TEST(OptimizerRegistryTest, GenericHandleRejectsUnknownType) {
  auto missing = tf::optimizer::optimizer("not_an_optimizer");

  EXPECT_TRUE(missing.type().empty());
  EXPECT_FALSE(missing.valid());
  EXPECT_FALSE(static_cast<bool>(missing));
  EXPECT_EQ(missing.id(), tf::optimizer::invalid_optimizer_id);
  EXPECT_TRUE(missing.parameterNames().empty());
  EXPECT_TRUE(missing.getParameter("anything").empty());

  // Invalid handles are safe no-ops.
  tf::tensor param;
  tf::tensor update;
  tf::tensor gradient;
  missing.create_parameter_update_graph(param, update, gradient);
  missing.execute();
  missing.setParameter(tf::optimizer::parameter::learning_rate, 0.1);
}

TEST(OptimizerRegistryTest, HandleExposesStringParameterNames) {
  auto sgd = tf::optimizer::optimizer(tf::optimizer::type::sgd);
  ASSERT_TRUE(sgd.valid());

  const std::unordered_set<std::string> names = sgd.parameterNames();
  EXPECT_TRUE(names.count(
      std::string(tf::optimizer::parameter::learning_rate)));
  EXPECT_TRUE(
      sgd.getParameter(tf::optimizer::parameter::learning_rate).empty());
}

TEST(OptimizerRegistryTest, UpdateGraphRoutingDoesNotStealTensorOwnership) {
  tf::tensor param;
  tf::tensor update;
  tf::tensor gradient;
  param.tf_create(tf_float64, 2, 2);
  gradient.tf_create(tf_float64, 2, 2);

  Tensor<std::float64_t> *param_ptr = param.getPtr();
  Tensor<std::float64_t> *gradient_ptr = gradient.getPtr();

  auto sgd = tf::optimizer::optimizer(tf::optimizer::type::sgd);
  sgd.create_parameter_update_graph(param, update, gradient);

  EXPECT_EQ(param.getPtr(), param_ptr);
  EXPECT_EQ(gradient.getPtr(), gradient_ptr);
  EXPECT_NE(update.getPtr(), nullptr);
}

TEST(OptimizerRegistryTest, CompileUsesStringOptimizerType) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  auto dense = tf::layer::dense(2);
  const std::vector<tf::tensor> dense_output = dense({x});
  tf::model model({x}, dense_output);

  model.compile(tf::optimizer::type::sgd, tf::loss::type::squared_error);
  EXPECT_TRUE(model.get_model_loss(dense_output[0]).valid());

  tf::model unknown_model({x}, dense_output);
  EXPECT_THROW(unknown_model.compile("not_an_optimizer",
                                     tf::loss::type::squared_error),
               std::invalid_argument);

  tf::optimizer::optimizer configured(tf::optimizer::type::sgd);
  tf::model configured_model({x}, dense_output);
  configured_model.compile(configured, tf::loss::type::squared_error);
  EXPECT_TRUE(configured_model.get_model_loss(dense_output[0]).valid());

  tf::optimizer::optimizer invalid;
  tf::model invalid_model({x}, dense_output);
  EXPECT_THROW(invalid_model.compile(invalid, tf::loss::type::squared_error),
               std::invalid_argument);
}

TEST(OptimizerRegistryTest, HandleDestructionReleasesRegistryId) {
  tf::optimizer::optimizer_id id = tf::optimizer::invalid_optimizer_id;
  {
    auto sgd = tf::optimizer::optimizer(tf::optimizer::type::sgd);
    ASSERT_TRUE(sgd.valid());
    id = sgd.id();
    ASSERT_TRUE(OptimizerRegistry::instance().contains(id));
  }

  EXPECT_NE(id, tf::optimizer::invalid_optimizer_id);
  EXPECT_FALSE(OptimizerRegistry::instance().contains(id));
}
