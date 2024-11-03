#pragma ONCE

namespace gpu
{
    __global__ void printData(double *a, unsigned x, unsigned y, unsigned z);

    __global__ void print(double *a);

    __global__ void cudaTranspose(unsigned int *a, unsigned int *b, int xsize, int ysize);

    __global__ void cudaDotMul(double *a, double *b, double *c, int x, int y, int a_m, int a_n, int b_m, int b_n);

    __global__ void cudaMatrixMulMultiParallesied(double *a, double *b, double *c, double *d, int a_m, int a_n, int b_m, int b_n);

    __global__ void cudaRollingSum(double *a);

    __device__ double cudaSubDotMul(double *a, double *b, int a_m, int a_n, int b_m, int b_n, int n);

    __global__ void cudaSubMul(double *a, double *b, double *d, int a_m, int a_n, int b_m, int b_n, int i, int j);

    __global__ void cudaMatrixMul(double *a, double *b, double *d, int a_m, int a_n, int b_m, int b_n, int i, int j);

    __global__ void matrixScalerMul(double* a, double b, double* c, unsigned x, unsigned y, unsigned z);

    __global__ void matrixAccuracyValue(double *confusion_matrix, double *accuracy, unsigned x, unsigned y);

    __global__ void matrixArgMax(double *a, double *b, unsigned x, unsigned y);

    __global__ void matrixDotMul(double *input_A, double *input_B, double *input_C, double *output, unsigned x, unsigned y, unsigned z);

    __global__ void matrixDifferentialParameters(double *input, double *delta_output, double *difference, double *d_weights_biases, unsigned x, unsigned y, unsigned z);

    __global__ void matrixDifferentialBiases(double *delta_output, double *difference, double *delta_biases, unsigned x, unsigned y);

    __global__ void matrixDifferentialInput(double *weights, double *delta_output, double *difference, double *delta_input, unsigned x, unsigned y, unsigned z);

    __global__ void matrixRollingSum(double *input, double *output, unsigned x, unsigned y, unsigned z);

    __global__ void matrixRelu(double *a, double *d_a, int x, int y);

    __global__ void matrixSigmoid(double *a, double *d_a, int x, int y);

    __global__ void matrixLinear(double *a, double *d_a, int x, int y);

    __global__ void matrixSoftmax(double *a, double *softmax_sum, double *d_a, unsigned x, unsigned y);

    __global__ void matrixSquaredError(double *a, double *b, unsigned x, unsigned y);

    __global__ void matrixSqrt(double *a, double *b, unsigned x, unsigned y);

    __global__ void matrixFindMean(double *a, unsigned x, unsigned y, unsigned mean);

    __global__ void matrixDifference(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y);

    __global__ void matrixCrossEntropy(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y, unsigned z);

    __global__ void matrixCrossEntropyDifference(double *, double *, double *, unsigned, unsigned, unsigned);

    __global__ void matrixConfusionMatrix(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y);

    __global__ void matrixBinaryCrossEntropy(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y);

    __global__ void matrixUpdateParameters(double *weights_biases, double *learning_rate, double *d_weights_biases, unsigned a_m, unsigned a_n, unsigned a_o);

    /// @brief Calculates weighted sum of input ndarray and output ndarray
    /// @tparam None
    /// @param input 2d array
    /// @param output 2d array
    /// @param x size of x dimension
    /// @param y size of y dimension
    /// @param alpha is weighted sum parameter
    /// @return output (call by referance)
    __global__ void maritrxWeightedSum(double *input, double *output, unsigned x, unsigned y, double alpha);

    /// @brief Normalize data of 2d input on y axis
    /// @tparam None
    /// @param input 2d array
    /// @param std_div 1d array in x dimension
    /// @param mean 1d array in x dimension
    /// @param output 2d array
    /// @param x size of x dimension
    /// @param y size of y dimension
    /// @return output (call by referance)
    __global__ void matrixNormalize(double *input, double *std_div, double *mean, double *output, unsigned x, unsigned y);

    /// @brief Applies scalling on normalized data on y axis
    /// @tparam None
    /// @param input 2d array
    /// @param gamma 1d array in x dimension, scaling factor
    /// @param beta 1d array in x dimension, bias factor
    /// @param output 2d array
    /// @param x size of x dimension
    /// @param y size of y dimension
    /// @return output (call by referance)
    __global__ void matrixNormalScaling(double *input, double *gamma, double *beta, double *output, unsigned x, unsigned y);

    __global__ void matrixExponentiallyWeightedMovingAvg(double , double *, double *, unsigned, unsigned, unsigned);

    __global__ void matrixUpdateWeightsBiasesRMSprop(double sigma, double epsalon, double *sum_d_weights_biases, double *d_weights_biases, unsigned a_m, unsigned a_n, unsigned a_o);

    __global__ void matrixUpdateWeightsBiasesADAM(double *sigma, double *epsalon, double *learning_rate, double *sum_d_weights_biases, double *sum_d_weights_biases_squared, double *d_weights_biases, double *weights_biases, unsigned a_m, unsigned a_n);

    __global__ void matrixUpdateLearningRateAdagrad(double, double , double *, double *, double *, unsigned, unsigned, unsigned);

    __global__ void matrixUpdateLearningRateAdadelta(double, double, double *, double *, double *, double *, unsigned, unsigned, unsigned);

}
