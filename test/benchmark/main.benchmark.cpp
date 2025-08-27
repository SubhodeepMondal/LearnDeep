// benchmarks/main.cpp

#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include <benchmark/benchmark.h>

int main(int argc, char **argv) {
  // Parse Abseil flags first (so --minloglevel etc. are handled)
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  //   absl::SetMinLogLevel(absl::LogSeverity::kInfo);

  // Hand remaining args to Google Benchmark
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
