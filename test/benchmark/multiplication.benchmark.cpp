#include <benchmark/benchmark.h>
#include <tensor.h>

// -------- Benchmark Eager Multiplication --------
static void mat_multiplication_eager_tensor(benchmark::State &state) {
  // Define two large matrices
  const int N = static_cast<int>(
      state.range(0)); // Size parameter (e.g. 256, 512, 1024 ...)
  tf::tensor A, B, C;
  A.tf_create(tf_float64, N, N);
  B.tf_create(tf_float64, N, N);
  C.tf_create(tf_float64, N, N);
  A.tensor_of(-0.5, 0.85);
  B.tensor_of(0.25, 0.75);

  for (auto _ : state) {
    C = A * B;                   // Perform matrix Multiplication
    benchmark::DoNotOptimize(C); // Prevent compiler optimization
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * N * N);
}

// Register this benchmark with different input sizes
BENCHMARK(mat_multiplication_eager_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12); // 4192 eleements
                    // ->Arg(1 << 14); // 16K elements

// -------- Benchmark Graph multiplication --------
static void mat_multiplication_graph_tensor(benchmark::State &state) {
  // Define two large matrices
  const int N = static_cast<int>(
      state.range(0)); // Size parameter (e.g. 256, 512, 1024 ...)
  tf::tensor A, B, C;
  A.tf_create(tf_float64, N, N);
  B.tf_create(tf_float64, N, N);
  C.tf_create(tf_float64, N, N);
  A.tensor_of(-0.5, 0.85);
  B.tensor_of(0.25, 0.75);

  tf::graph g_multiplication;
  g_multiplication.tf_create_graph();

  C = A.mul(g_multiplication, B);

  for (auto _ : state) {
    g_multiplication.graph_execute(); // Perform matrix multiplication
    benchmark::DoNotOptimize(C);      // Prevent compiler optimization
  }
  g_multiplication.graph_clear();


  state.SetItemsProcessed(int64_t(state.iterations()) * N * N);
}

// Register this benchmark with different input sizes
BENCHMARK(mat_multiplication_graph_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12); // 4192 eleements
                    // ->Arg(1 << 14); // 16K elements
