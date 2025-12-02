#include <benchmark/benchmark.h>
#include <tensor.h>

// -------- Benchmark Graph sigmoid --------
static void mat_sigmoid_graph_tensor(benchmark::State &state) {
  // Define two large matrices
  const int N = static_cast<int>(
      state.range(0)); // Size parameter (e.g. 256, 512, 1024 ...)
  tf::tensor A, C;
  A.tf_create(tf_float64, N, N);
  C.tf_create(tf_float64, N, N);
  A.tensor_of(-0.5, 0.85);
  {
    tf::graph_context ctx;

    C = A.sigmoid(0);

    for (auto _ : state) {
      ctx.run();                   // Perform matrix sigmoid
      benchmark::DoNotOptimize(C); // Prevent compiler optimization
    }
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * N * N);
}

// Register this benchmark with different input sizes

// Register this benchmark with different input sizes
#ifdef CUDA_ENABLED
BENCHMARK(mat_sigmoid_graph_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12)  // 4192 eleements
    ->Arg(1 << 14); // 16K elements
#else
BENCHMARK(mat_sigmoid_graph_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12)  // 4192 eleements
    ->Arg(1 << 14)  // 16K elements
    ->Arg(1 << 16); // 16K elements
#endif
