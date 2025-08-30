#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/log/globals.h" 
#include <gtest/gtest.h>

int main(int argc, char** argv) {

  // absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  // absl::SetMinLogLevel(absl::LogSeverity::kInfo);
  // absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}