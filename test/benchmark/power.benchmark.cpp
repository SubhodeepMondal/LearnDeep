#include <benchmark/benchmark.h>
#include <tensor.h>

// -------- Benchmark Eager Power --------
static void mat_power_eager_tensor(benchmark::State &state) {
  // Define two large matrices
  const int N = static_cast<int>(
      state.range(0)); // Size parameter (e.g. 256, 512, 1024 ...)
  tf::tensor A,C;
  A.tf_create(tf_float64, N, N);
  C.tf_create(tf_float64, N, N);
  A.tensor_of(-0.5, 0.85);

  for (auto _ : state) {
    C = A.pow(2);                   // Perform matrix Power
    benchmark::DoNotOptimize(C); // Prevent compiler optimization
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * N * N);
}

// Register this benchmark with different input sizes
BENCHMARK(mat_power_eager_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12); // 4192 eleements
                    // ->Arg(1 << 14); // 16K elements

// -------- Benchmark Graph power --------
static void mat_power_graph_tensor(benchmark::State &state) {
  // Define two large matrices
  const int N = static_cast<int>(
      state.range(0)); // Size parameter (e.g. 256, 512, 1024 ...)
  tf::tensor A, C;
  A.tf_create(tf_float64, N, N);
  C.tf_create(tf_float64, N, N);
  A.tensor_of(-0.5, 0.85);

  tf::graph g_power;
  g_power.tf_create_graph();

  C = A.pow(g_power, 2);

  for (auto _ : state) {
    g_power.graph_execute(); // Perform matrix power
    benchmark::DoNotOptimize(C);      // Prevent compiler optimization
  }

  g_power.graph_clear();

  state.SetItemsProcessed(int64_t(state.iterations()) * N * N);
}

// Register this benchmark with different input sizes
BENCHMARK(mat_power_graph_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12); // 4192 eleements
                    // ->Arg(1 << 14); // 16K elements
