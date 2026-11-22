#include <LearnDeep/api/tensor.h>
#include <callback/callback.hpp>
#include <gtest/gtest.h>
#include <losses/loss_registry.hpp>

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

TEST(LossRegistryTest, ContainsBuiltInLosses) {
  const std::unordered_set<std::string> &names =
      LossRegistry::instance().names();

  EXPECT_TRUE(names.count(std::string(tf::loss::type::squared_error)));
  EXPECT_TRUE(names.count(
      std::string(tf::loss::type::categorical_cross_entropy)));
}

TEST(LossRegistryTest, GenericHandleSupportsRegistryConstruction) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  LossConfig config;
  config.set(std::string(tf::loss::parameter::inputs),
             std::vector<Tensor<std::float64_t> *>{x.getPtr()});
  config.set(std::string(tf::loss::parameter::name), std::string{"se_from_config"});

  auto se = tf::loss::loss(tf::loss::type::squared_error, config);
  ASSERT_TRUE(se.valid());
  EXPECT_TRUE(static_cast<bool>(se));
  EXPECT_EQ(se.type(), tf::loss::type::squared_error);
  EXPECT_EQ(se.get_loss_name(), "se_from_config");
  EXPECT_NE(se.id(), tf::loss::invalid_loss_id);
  EXPECT_EQ(se.get_no_of_inputs(), 1u);

  auto se_copy = se;
  EXPECT_EQ(se_copy.id(), se.id());
  EXPECT_TRUE(LossRegistry::instance().contains(se.id()));
}

TEST(LossRegistryTest, GenericHandleRejectsUnknownType) {
  auto missing = tf::loss::loss("not_a_loss");

  EXPECT_TRUE(missing.type().empty());
  EXPECT_FALSE(missing.valid());
  EXPECT_FALSE(static_cast<bool>(missing));
  EXPECT_EQ(missing.id(), tf::loss::invalid_loss_id);
  EXPECT_TRUE(missing.parameterNames().empty());
  EXPECT_TRUE(missing.getParameter("anything").empty());
}

TEST(LossRegistryTest, HandleExposesStringParameterNames) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  auto se = tf::loss::loss(tf::loss::type::squared_error,
                           makeLossConfig({x.getPtr()}, "se_parameters"));
  ASSERT_TRUE(se.valid());

  const std::unordered_set<std::string> names = se.parameterNames();
  EXPECT_TRUE(names.count(
      std::string(tf::loss::parameter::squared_error_predicted_output)));
  EXPECT_TRUE(names.count(
      std::string(tf::loss::parameter::squared_error_target_output)));
  EXPECT_TRUE(names.count(
      std::string(tf::loss::parameter::squared_error_grad_predicted_output)));
  EXPECT_TRUE(
      se.getParameter(tf::loss::parameter::squared_error_predicted_output)
          .empty());
}

TEST(LossRegistryTest, TypedWrappersUseRegistry) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  tf::loss::squared_error se({x.getPtr()}, "se_wrapper");
  ASSERT_TRUE(se.valid());
  EXPECT_EQ(se.type(), tf::loss::type::squared_error);

  tf::loss::categorical_cross_entropy cce({x.getPtr()}, "cce_wrapper");
  ASSERT_TRUE(cce.valid());
  EXPECT_EQ(cce.type(), tf::loss::type::categorical_cross_entropy);
}

TEST(LossRegistryTest, CompileUsesStringLossType) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  auto dense = tf::layer::dense(2);
  const std::vector<tf::tensor> dense_output = dense({x});
  tf::model model({x}, dense_output);

  model.compile(OptimizerType::SGD, tf::loss::type::squared_error);
  auto loss = model.get_model_loss(dense_output[0]);
  ASSERT_TRUE(loss.valid());
  // The historical global uniquifier may append a suffix when other tests
  // already created a loss with this name.
  EXPECT_EQ(loss.get_loss_name().rfind("mean_squared_loss_1", 0), 0u);

  tf::model unknown_model({x}, dense_output);
  EXPECT_THROW(unknown_model.compile(OptimizerType::SGD, "not_a_loss"),
               std::invalid_argument);
}

TEST(LossRegistryTest, StaleLossCallbackIdsDoNotBlockTraining) {
  tf::tensor x;
  x.tf_create(tf_float64, 2, 2);
  x.tensor_of(0.0, 1.0);

  tf::callback::trace trace;
  tf::loss::loss_id stale_id = tf::loss::invalid_loss_id;
  {
    auto dead = tf::loss::loss(
        tf::loss::type::squared_error,
        makeLossConfig({x.getPtr()}, "stale_loss"));
    stale_id = dead.id();
    trace.record_scalar_loss_on_batch_end(dead);
  }
  ASSERT_NE(stale_id, tf::loss::invalid_loss_id);
  ASSERT_FALSE(LossRegistry::instance().contains(stale_id));

  auto dense = tf::layer::dense(3);
  const std::vector<tf::tensor> dense_output = dense({x});
  tf::model model({x}, dense_output);
  model.compile(OptimizerType::SGD, tf::loss::type::squared_error);
  EXPECT_TRUE(trace.callback()->callOnTrainingBegin(model));

  tf::tensor target;
  target.tf_create(tf_float64, 3, 2);
  target.tensor_of(0.0, 0.0);
  model.shuffle(false);
  model.fit({x}, {target}, {trace.callback()}, 1, 2);
}
