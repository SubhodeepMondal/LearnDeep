#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>
#include <losses/loss_registry.hpp>

#include <string>
#include <unordered_set>
#include <vector>

TEST(LossRegistryTest, ContainsBuiltInLosses) {
  const std::unordered_set<std::string> &names =
      LossRegistry::instance().names();

  EXPECT_TRUE(names.count("squared_error"));
  EXPECT_TRUE(names.count("categorical_cross_entropy"));
}

TEST(LossRegistryTest, GenericHandleSupportsRegistryConstruction) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  LossConfig config;
  config.set(std::string(tf::loss::parameter::inputs),
             std::vector<Tensor<std::float64_t> *>{x.getPtr()});
  config.set(std::string(tf::loss::parameter::name), std::string{"se_from_config"});

  tf::loss::loss se(tf::loss::type::squared_error, config);
  ASSERT_TRUE(se.valid());
  EXPECT_TRUE(static_cast<bool>(se));
  EXPECT_EQ(se.type(), tf::loss::type::squared_error);
  EXPECT_EQ(se.get_loss_name(), "se_from_config");
  EXPECT_NE(se.id(), tf::loss::invalid_loss_id);
  EXPECT_EQ(se.get_no_of_inputs(), 1u);

  tf::loss::loss se_copy = se;
  EXPECT_EQ(se_copy.id(), se.id());
  EXPECT_EQ(se_copy.get_loss_ptr(), se.get_loss_ptr());
}

TEST(LossRegistryTest, GenericHandleRejectsUnknownType) {
  tf::loss::loss missing("not_a_loss");

  EXPECT_TRUE(missing.type().empty());
  EXPECT_FALSE(missing.valid());
  EXPECT_FALSE(static_cast<bool>(missing));
  EXPECT_EQ(missing.id(), tf::loss::invalid_loss_id);
}

TEST(LossRegistryTest, LegacyEnumConstructorUsesRegistry) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  tf::loss::loss se(LossType::squared_error, {x.getPtr()}, "se_from_enum");
  ASSERT_TRUE(se.valid());
  EXPECT_EQ(se.type(), tf::loss::type::squared_error);
  EXPECT_EQ(se.get_loss_name(), "se_from_enum");

  tf::loss::loss cce(LossType::categorical_cross_entropy, {x.getPtr()},
                     "cce_from_enum");
  ASSERT_TRUE(cce.valid());
  EXPECT_EQ(cce.type(), tf::loss::type::categorical_cross_entropy);
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
