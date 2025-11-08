#ifndef GPU_MICRO_KERNELS_CUH
#define GPU_MICRO_kERNELS_CUH
#include <cuda_runtime.h>

namespace gpu_kernel {

__global__ void printData(double *a, unsigned x, unsigned y, unsigned z);

__global__ void print(double *a);

__global__ void cudaDotMul(double *a, double *b, double *c, int x, int y,
                           int a_m, int a_n, int b_m, int b_n);

__global__ void cudaMatrixMulMultiParallesied(double *a, double *b, double *c,
                                              double *d, int a_m, int a_n,
                                              int b_m, int b_n);

__global__ void cudaRollingSum(double *a);

__device__ double cudaSubDotMul(double *a, double *b, int a_m, int a_n, int b_m,
                                int b_n, int n);

__global__ void cudaSubMul(double *a, double *b, double *d, int a_m, int a_n,
                           int b_m, int b_n, int i, int j);

__global__ void cudaMatrixMul(double *a, double *b, double *d, int a_m, int a_n,
                              int b_m, int b_n, int i, int j);

__global__ void matrixSum(double *a, double *b, double *c, unsigned x,
                          unsigned y);

__global__ void matrixHadamardMul(double *a, double *b, double *c, unsigned x,
                                  unsigned y);

__global__ void matrixMul(double *a, double *b, double *c, unsigned x,
                          unsigned y, unsigned z);

__global__ void matrixScalerMul(double *input, double scaler_value,
                                double *output, unsigned x, unsigned y);

__global__ void matrixAccuracyValue(double *confusion_matrix, double *accuracy,
                                    unsigned x, unsigned y);

__global__ void matrixArgMax(double *a, double *b, unsigned x, unsigned y);

__global__ void matrixDotMul(double *input_A, double *input_B, double *input_C,
                             double *output, unsigned x, unsigned y,
                             unsigned z);

__global__ void matrixDifferentialParameters(double *input,
                                             double *delta_output,
                                             double *difference,
                                             double *d_parameters, unsigned x,
                                             unsigned y, unsigned z);

__global__ void matrixDifferentialBiases(double *delta_output,
                                         double *difference,
                                         double *delta_biases, unsigned x,
                                         unsigned y);

__global__ void matrixDifferentialInput(double *weights, double *delta_output,
                                        double *difference, double *delta_input,
                                        unsigned x, unsigned y, unsigned z);

__global__ void matrixRollingSum(double *input, double *output, unsigned x,
                                 unsigned y, unsigned z);

__global__ void matrixRelu(double *a, double *d_a, int x, int y);

__global__ void matrixSigmoid(double *a, double *d_a, int x, int y);

__global__ void matrixSub(double *input_A, double *input_B, double *output,
                          unsigned x, unsigned y);

__global__ void matrixLinear(double *a, double *d_a, int x, int y);

__global__ void matrixSoftmax(double *a, double *softmax_sum, double *d_a,
                              unsigned x, unsigned y);

__global__ void matrixSquaredError(double *a, double *b, unsigned x,
                                   unsigned y);

__global__ void matrixSqrt(double *a, double *b, unsigned x, unsigned y);

__global__ void matrixFindMean(double *a, unsigned x, unsigned y,
                               unsigned mean);

__global__ void matrixDifference(double *input_A, double *input_B,
                                 double *output_C, unsigned x, unsigned y);

__global__ void matrixTranspose(double *input_A, double *input_B, unsigned x,
                                unsigned y);

} // namespace gpu_kernel
#endif // GPU_MICRO_KERNEL_CUH