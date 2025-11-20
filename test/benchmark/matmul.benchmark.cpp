#include <benchmark/benchmark.h>
#include <tensor.h>

// -------- Benchmark Eager Matmul --------
static void mat_matmul_eager_tensor(benchmark::State &state) {
  // Define two large matrices
  const int N = static_cast<int>(
      state.range(0)); // Size parameter (e.g. 256, 512, 1024 ...)
  tf::tensor A, B, C;
  A.tf_create(tf_float64, N, N);
  B.tf_create(tf_float64, N, N);
  C.tf_create(tf_float64, N, N);
  A.tensor_of(-0.5, 0.85);
  B.tensor_of(0.25, 0.75);

  // Initialize matrices with some values
  // for (int i = 0; i < N * N; ++i) {
  //   A.getPtr()[i] = static_cast<double>(i);
  //   B.getPtr()[i] = static_cast<double>(i);
  // }

  for (auto _ : state) {
    C = A.matmul(B);             // Perform matrix Matmul
    benchmark::DoNotOptimize(C); // Prevent compiler optimization
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * N * N);
}
// Register this benchmark with different input sizes
#ifdef CUDA_ENABLED
BENCHMARK(mat_matmul_eager_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12)  // 4192 eleements
    ->Arg(1 << 14); // 16K elements
#else
BENCHMARK(mat_matmul_eager_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12)  // 4192 eleements
    ->Arg(1 << 14); // 16K elements
#endif

// -------- Benchmark Graph Matmul --------
static void mat_matmul_graph_tensor(benchmark::State &state) {
  // Define two large matrices
  const int N = static_cast<int>(
      state.range(0)); // Size parameter (e.g. 256, 512, 1024 ...)
  tf::tensor A, B, C;
  A.tf_create(tf_float64, N, N);
  B.tf_create(tf_float64, N, N);
  C.tf_create(tf_float64, N, N);
  A.tensor_of(-0.5, 0.85);
  B.tensor_of(0.25, 0.75);

  tf::graph g_matmul;
  g_matmul.tf_create_graph();

  C = A.matmul(g_matmul, B);

  for (auto _ : state) {
    g_matmul.graph_execute();    // Perform matrix Matmul
    benchmark::DoNotOptimize(C); // Prevent compiler optimization
  }
  g_matmul.graph_clear();

  state.SetItemsProcessed(int64_t(state.iterations()) * N * N);
}

// Register this benchmark with different input sizes

// Register this benchmark with different input sizes
#ifdef CUDA_ENABLED
BENCHMARK(mat_matmul_graph_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12)  // 4192 eleements
    ->Arg(1 << 14); // 16K elements
#else
BENCHMARK(mat_matmul_graph_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12)  // 4192 eleements
    ->Arg(1 << 14); // 16K elements
#endif
