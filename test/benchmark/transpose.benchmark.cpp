#include <benchmark/benchmark.h>
#include <tensor.h>

// ------------------  BenchMark Eager Transpose ---------------
static void mat_transpose_eager_tensor(benchmark::State &state) {
  const int N = static_cast<int>(
      state.range(0)); // Size parameter (e.g. 256, 512, 1024 ...)

  tf::tensor A, B, C;
  A.tf_create(tf_float64, N, N);
  C.tf_create(tf_float64, N, N);
  A.tensor_of(-0.5, 0.85);

  for (auto _ : state) {
    C = A.transpose();           // Perform matrix addition
    benchmark::DoNotOptimize(C); // Prevent compiler optimization
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * N * N);
}
// Register this benchmark with different input sizes
BENCHMARK(mat_transpose_eager_tensor)
    ->Arg(1 << 8)   // 256 elements
    ->Arg(1 << 9)   // 512 elements
    ->Arg(1 << 10)  // 1024 elements
    ->Arg(1 << 12)  // 4192 eleements
    ->Arg(1 << 14); // 16K elements