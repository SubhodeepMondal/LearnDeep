#include <LearnDeep/api/tensor.h>
#include <callback/callback.hpp>
#include <gtest/gtest.h>
#include <losses/loss_registry.hpp>
#include <optimizers/optimizers.hpp>
#include <optimizers/optimizer_registry.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

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

TEST(OptimizerRegistryTest, SetParameterUsesCanonicalTypeForRegistryAliases) {
  ASSERT_TRUE(OptimizerRegistry::instance().registerOptimizer(
      "sgd_alias_for_set_parameter",
      [](const OptimizerConfig &) -> std::unique_ptr<Optimizer> {
        return std::make_unique<SGD>();
      }));

  auto alias = tf::optimizer::optimizer("sgd_alias_for_set_parameter");
  ASSERT_TRUE(alias.valid());
  EXPECT_EQ(alias.type(), tf::optimizer::type::sgd);

  // The alias has no setter of its own, so this only succeeds when
  // setParameter() dispatches on the optimizer's canonical type.
  alias.setParameter(tf::optimizer::parameter::learning_rate, 0.125);

  auto locked = OptimizerRegistry::instance().lock(alias.id());
  ASSERT_NE(locked, nullptr);
  auto *sgd = dynamic_cast<SGD *>(locked.get());
  ASSERT_NE(sgd, nullptr);
  EXPECT_DOUBLE_EQ(sgd->learningRate(), 0.125);
}

TEST(OptimizerRegistryTest, OwnershipIsSharedByModelAndApiHandle) {
  tf::tensor x;
  x.tf_create(tf_float64, 2, 2);

  std::unique_ptr<tf::model> model;
  tf::optimizer::optimizer_id id = tf::optimizer::invalid_optimizer_id;
  {
    auto sgd = tf::optimizer::optimizer(tf::optimizer::type::sgd);
    ASSERT_TRUE(sgd.valid());
    id = sgd.id();

    tf::layer::dense dense(2);
    const std::vector<tf::tensor> outputs = dense({x});
    model = std::make_unique<tf::model>(std::vector<tf::tensor>{x}, outputs);
    model->compile(sgd, tf::loss::type::squared_error);
    EXPECT_TRUE(OptimizerRegistry::instance().contains(id));
  }

  ASSERT_NE(id, tf::optimizer::invalid_optimizer_id);
  ASSERT_TRUE(OptimizerRegistry::instance().contains(id));

  model.reset();
  EXPECT_FALSE(OptimizerRegistry::instance().contains(id));
}

TEST(OptimizerRegistryTest, RepeatedCompileReleasesStaleOptimizerAndLosses) {
  tf::tensor x;
  x.tf_create(tf_float64, 2, 2);
  x.tensor_of(0.0, 1.0);

  tf::tensor target;
  target.tf_create(tf_float64, 2, 2);
  target.tensor_of(0.0, 0.0);

  tf::layer::dense dense(2);
  const std::vector<tf::tensor> outputs = dense({x});
  auto model = std::make_unique<tf::model>(std::vector<tf::tensor>{x}, outputs);
  model->shuffle(false);

  tf::optimizer::optimizer_id first_id = tf::optimizer::invalid_optimizer_id;
  std::vector<tf::loss::loss_id> first_losses;
  {
    auto first = tf::optimizer::optimizer(tf::optimizer::type::sgd);
    ASSERT_TRUE(first.valid());
    first_id = first.id();
    model->compile(first, tf::loss::type::squared_error);
  }
  first_losses = model->get_losses();
  ASSERT_EQ(first_losses.size(), outputs.size());
  ASSERT_TRUE(OptimizerRegistry::instance().contains(first_id));
  for (tf::loss::loss_id loss_id : first_losses)
    ASSERT_TRUE(LossRegistry::instance().contains(loss_id));

  // Train once so one-shot gradient flags are set before recompiling.
  model->fit({x}, {target}, {}, 1, 2);

  tf::optimizer::optimizer_id second_id = tf::optimizer::invalid_optimizer_id;
  {
    auto second = tf::optimizer::optimizer(tf::optimizer::type::sgd);
    ASSERT_TRUE(second.valid());
    second_id = second.id();
    model->compile(second, tf::loss::type::squared_error);
  }

  EXPECT_NE(second_id, first_id);
  EXPECT_FALSE(OptimizerRegistry::instance().contains(first_id));
  EXPECT_TRUE(OptimizerRegistry::instance().contains(second_id));
  EXPECT_EQ(model->get_losses().size(), first_losses.size());
  for (tf::loss::loss_id loss_id : first_losses)
    EXPECT_FALSE(LossRegistry::instance().contains(loss_id));

  // The reset gradient flags must let the new graph record a fresh update.
  tf::callback::trace trace;
  trace.record_parameter_on_batch_end(dense, "updated_weight", false);
  model->fit({x}, {target}, {trace.callback()}, 1, 2);
  EXPECT_EQ(trace.get_parameter_on_batch_end(dense, "updated_weight").size(),
            1u);
}

TEST(OptimizerRegistryTest, UpdateGraphProducesExpectedWeightAndBiasValues) {
  tf::optimizer::config config;
  config.set(std::string(tf::optimizer::parameter::learning_rate), 0.5);
  auto sgd = tf::optimizer::optimizer(tf::optimizer::type::sgd, config);
  ASSERT_TRUE(sgd.valid());

  std::float64_t weight_param_data[] = {1.0, 2.0, 3.0, 4.0};
  std::float64_t weight_gradient_data[] = {0.2, 0.4, 0.6, 0.8};
  tf::tensor weight_param;
  tf::tensor weight_gradient;
  tf::tensor updated_weight;
  weight_param.tf_create(tf_float64, 2, 2);
  weight_gradient.tf_create(tf_float64, 2, 2);
  weight_param.tensor_of(weight_param_data);
  weight_gradient.tensor_of(weight_gradient_data);
  sgd.create_parameter_update_graph(weight_param, updated_weight,
                                    weight_gradient);

  std::float64_t bias_param_data[] = {5.0, 6.0};
  std::float64_t bias_gradient_data[] = {0.1, 0.2};
  tf::tensor bias_param;
  tf::tensor bias_gradient;
  tf::tensor updated_bias;
  bias_param.tf_create(tf_float64, 2, 1);
  bias_gradient.tf_create(tf_float64, 2, 1);
  bias_param.tensor_of(bias_param_data);
  bias_gradient.tensor_of(bias_gradient_data);
  sgd.create_parameter_update_graph(bias_param, updated_bias, bias_gradient);

  sgd.execute();
  ASSERT_NE(updated_weight.getPtr(), nullptr);
  ASSERT_NE(updated_bias.getPtr(), nullptr);

  for (unsigned i = 0; i < 4; i++)
    EXPECT_NEAR(updated_weight.getData()[i],
                weight_param_data[i] - 0.5 * weight_gradient_data[i], 1e-12);
  for (unsigned i = 0; i < 2; i++)
    EXPECT_NEAR(updated_bias.getData()[i],
                bias_param_data[i] - 0.5 * bias_gradient_data[i], 1e-12);
}
