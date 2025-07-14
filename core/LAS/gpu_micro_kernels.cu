#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_micro_kernels.cuh"

__global__ void gpu::printData(double *a, unsigned x, unsigned y, unsigned z)
{
    int i, j, k;
    for (i = 0; i < z; i++)
        for (j = 0; j < y; j++)
            for (k = 0; k < x; k++)
            {
                if (k == x - 1)
                    if (j == y - 1)
                        printf(" %lf\n\n", a[k + j * x + i * x * y]);
                    else
                        printf(" %lf\n", a[k + j * x + i * x * y]);
                else
                    printf(" %lf", a[k + j * x + i * x * y]);
            }
}

__global__ void gpu::print(double *a)
{
    printf("%.6lf", *(a));
}

__global__ void gpu::cudaTranspose(unsigned int *a, unsigned int *b, int xsize, int ysize)
{
    int ix, iy, mat_in, mat_tra;

    __shared__ unsigned int smallblock[32][32];

    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    mat_in = ix * xsize + iy;

    int bidx, icol, irow;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    mat_tra = iy * ysize + ix;

    smallblock[threadIdx.x][threadIdx.y] = mat_in;
    __syncthreads();
    b[mat_tra] = smallblock[icol][irow];
}

__global__ void gpu::cudaDotMul(double *a, double *b, double *c, int x, int y, int a_m, int a_n, int b_m, int b_n)
{
    int n = b_m;
    int ind, i;
    double val;
    unsigned mask = 0xffffffff;

    ind = (threadIdx.x + x * b_m) + (y * a_m * b_m);
    if (threadIdx.x < a_n)
    {
        c[ind] = a[threadIdx.x + x * a_n] * b[threadIdx.x * b_n + y];
    }
    else
        c[ind] = 0.0f;

    __syncthreads();

    if (threadIdx.x < a_n)
    {
        for (i = n / 2; i > 0; i /= 2)
        {
            val = c[ind];
            val = __shfl_down_sync(mask, val, i);
            c[ind] += val;
        }
    }
}

__global__ void gpu::cudaMatrixMulMultiParallesied(double *a, double *b, double *c, double *d, int a_m, int a_n, int b_m, int b_n)
{
    int ix, iy, rowDim;
    ix = threadIdx.x + blockIdx.x * blockDim.x;
    iy = threadIdx.y + blockIdx.y * blockDim.y;
    rowDim = b_m;
    d[ix + iy * a_m] = 0;
    if (ix < a_m && iy < b_n)
    {
        cudaDotMul<<<1, rowDim>>>(a, b, c, ix, iy, a_m, a_n, b_m, b_n);
        d[ix + iy * a_m] += c[ix * b_m * a_m + iy * b_m];
    }
}

__global__ void gpu::cudaRollingSum(double *a)
{
    int n = blockDim.x;
    int ind, i, k;
    double val;
    ind = threadIdx.x;
    unsigned x = 0xffffffff;

    k = n;
    if (threadIdx.x < n)
    {
        for (i = n / 2; i > 0; i /= 2)
        {
            val = a[ind];
            val = __shfl_down_sync(x, val, i);
            a[ind] += val;
            k /= 2;
        }
    }
}

__device__ double gpu::cudaSubDotMul(double *a, double *b, int a_m, int a_n, int b_m, int b_n, int n)
{
    double sum = 0;

    for (int i = 0; i < n; i++)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

__global__ void gpu::cudaSubMul(double *a, double *b, double *d, int a_m, int a_n, int b_m, int b_n, int i, int j)
{
    __shared__ double Y_shared[32][33];

    double val;

    int Ai, Bi, Bj, n;

    Bi = j + threadIdx.x;
    Bj = i + threadIdx.y;
    Ai = i + (threadIdx.y + blockDim.y * blockIdx.y) * a_n;
    (a_n - i) >= 32 ? n = 32 : n = (a_n - i);

    if (Bi < b_n && Bj < b_m)
    {
        Y_shared[threadIdx.x][threadIdx.y] = b[Bi + Bj * b_n];
    }
    __syncthreads();

    if (Bi < b_n && (threadIdx.y + blockDim.y * blockIdx.y) < a_m)
    {

        val = cudaSubDotMul((a + Ai), Y_shared[threadIdx.x], a_m, a_n, b_m, b_n, n);
        __syncthreads();

        d[Bi + (threadIdx.y + blockDim.y * blockIdx.y) * b_n] += val;
    }
}

__global__ void gpu::cudaMatrixMul(double *a, double *b, double *d, int a_m, int a_n, int b_m, int b_n, int i, int j)
{
    __shared__ double Y_shared[32][33];
    __shared__ double X_shared[32][32];

    double val;

    int Ai, Bi, Bj, n, idx_Ai;

    Bi = j + threadIdx.x;
    Bj = i + threadIdx.y;
    Ai = i + (threadIdx.y + blockDim.y * blockIdx.y) * a_n;
    idx_Ai = (threadIdx.y + blockDim.y * blockIdx.y);

    (a_n - i) >= 32 ? n = 32 : n = (a_n - i);

    if (Bi < b_n && Bj < b_m)
    {
        Y_shared[threadIdx.x][threadIdx.y] = b[Bi + Bj * b_n];
    }
    if (idx_Ai < a_m && threadIdx.x < a_n)
    {
        X_shared[threadIdx.y][threadIdx.x] = a[Ai + threadIdx.x];
    }
    __syncthreads();

    if (Bi < b_n && (threadIdx.y + blockDim.y * blockIdx.y) < a_m)
    {

        val = cudaSubDotMul(X_shared[threadIdx.y], Y_shared[threadIdx.x], a_m, a_n, b_m, b_n, n);
        __syncthreads();

        d[Bi + (threadIdx.y + blockDim.y * blockIdx.y) * b_n] += val;
    }
}

__global__ void gpu::matrixSum(double *a, double *b, double *c, unsigned x, unsigned y){
    unsigned id_x, id_y, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);
    lin_idx = id_x + id_y * x;

    if (id_x < x && id_y < y)
        c[lin_idx] = a[lin_idx] + b[lin_idx];
}

__global__ void gpu::matrixScalerMul(double *input, double scaler_value, double *output, unsigned x, unsigned y, unsigned z)
{

    unsigned id_x, id_y, id_z, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);
    id_z = threadIdx.z + (blockDim.z * blockIdx.z);
    lin_idx = id_x + id_y * x + id_z * x * y;

    if (id_x < x && id_y < y && id_z < z)
        output[lin_idx] = scaler_value * input[lin_idx];
}

__global__ void gpu::matrixAccuracyValue(double *confusion_matrix, double *accuracy, unsigned x, unsigned y)
{
    unsigned i, true_values = 0;

    for (i = 0; i < x; i++)
        true_values += confusion_matrix[i + i * x];

    *accuracy = (double)true_values / y;
}

__global__ void gpu::matrixArgMax(double *a, double *b, unsigned x, unsigned y)
{
    unsigned i, intr_y, index;
    double large = -1.0;

    intr_y = threadIdx.y + (blockIdx.y * blockDim.y);

    if (intr_y < y)
    {
        index = 0;
        for (i = 0; i < x; i++)
            if (large < a[i + intr_y * x])
            {
                index = i;
                large = a[i + intr_y * x];
            }

        for (i = 0; i < x; i++)
            b[i + intr_y * x] = (index == i) ? 1 : 0;
    }
}

__global__ void gpu::matrixDotMul(double *input_A, double *input_B, double *input_C, double *output, unsigned x, unsigned y, unsigned z)
{
    // m : features, n : neurons;
    // x : max feature, y : max neuron;
    unsigned intr_x, intr_y, intr_z, inp_lin, w_lin, res_lin;

    intr_x = threadIdx.x + (blockIdx.x * blockDim.x); // neuron axis
    intr_y = threadIdx.y + (blockIdx.y * blockDim.y); // batch axis
    intr_z = threadIdx.z + (blockIdx.z * blockDim.z); // batch axis

    inp_lin = intr_z + (intr_y * z);                    // input linear index.
    w_lin = intr_x + (intr_z * x);                      // z = features + 1 for bias.
    res_lin = intr_x + (intr_y * x) + (intr_z * x * y); // resluting array input index.

    if (intr_x < x && intr_y < y && intr_z < z)
    {
        output[res_lin] = input_A[inp_lin] * input_B[w_lin]; // (double)
        // c[res_lin] = a[inp_lin];
    }
    else if (intr_x < x && intr_y < y && intr_z == z)
    {
        output[res_lin] = input_C[intr_x];
    }
}

__global__ void gpu::matrixDifferentialParameters(double *input, double *delta_output, double *difference, double *d_parameters, unsigned x, unsigned y, unsigned z)
{
    unsigned indx_x, indx_y, indx_z, out_lin, inp_lin, diff_lin;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;
    indx_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (indx_x < x && indx_y < y && indx_z < z)
    {
        out_lin = indx_x + indx_y * x + indx_z * x * y;
        inp_lin = indx_y + indx_z * y;
        diff_lin = indx_x + indx_z * x;

        d_parameters[out_lin] = delta_output[diff_lin] * difference[diff_lin];
        if (input)
            d_parameters[out_lin] *= input[inp_lin];
    }
}

__global__ void gpu::matrixDifferentialBiases(double *delta_output, double *difference, double *delta_biases, unsigned x, unsigned y)
{
    unsigned indx_x, indx_y, out_lin, diff_lin;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (indx_x < x && indx_y < y)
    {
        out_lin = indx_x + indx_y * x;
        diff_lin = indx_x + indx_y * x;

        delta_biases[out_lin] = 2 * delta_output[diff_lin] * difference[diff_lin]; //
    }
}

__global__ void gpu::matrixDifferentialInput(double *weights, double *delta_output, double *difference, double *delta_input, unsigned x, unsigned y, unsigned z)
{
    unsigned indx_x, indx_y, indx_z, out_lin, inp_lin, diff_lin;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;
    indx_z = threadIdx.z + blockIdx.z * blockDim.z;
    out_lin = indx_x + indx_y * x + indx_z * x * y;

    if (indx_x < x && indx_y < y & indx_z < z)
    {
        out_lin = indx_x + indx_y * x + indx_z * x * y;
        diff_lin = indx_y + indx_z * y;
        inp_lin = indx_x + indx_z * x;

        delta_input[out_lin] = 2 * weights[inp_lin] * difference[diff_lin] * delta_output[diff_lin]; //
    }
}

__global__ void gpu::matrixRollingSum(double *input, double *output, unsigned x, unsigned y, unsigned z)
{
    // input dimension: xyz, (z is adding axis, xy is bubble up axises).
    // output dimension: xy

    // x: neurons, y: features
    unsigned intr_x, intr_y, inp_lin, out_lin, i;
    double val;

    intr_x = threadIdx.x + (blockIdx.x * blockDim.x);
    intr_y = threadIdx.y + (blockIdx.y * blockDim.y);
    out_lin = intr_x + intr_y * x;

    if (intr_x < x && intr_y < y)
    {
        val = 0.0;
        for (i = 0; i < z; i++)
        {
            inp_lin = out_lin + i * x * y;
            val += input[inp_lin];
        }
        output[out_lin] = val;
    }
}

__global__ void gpu::matrixRelu(double *a, double *d_a, int x, int y)
{
    // x: neuron, y: feature. m: max_neuron.
    unsigned id_x, id_y, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);

    if (id_x < x && id_y < y)
    {
        lin_idx = id_x + id_y * x;
        if (a[lin_idx] > 0)
        {
            d_a[lin_idx] = 1;
        }
        else
        {

            a[lin_idx] = 0;
            d_a[lin_idx] = 0;
        }
    }
}

__global__ void gpu::matrixSigmoid(double *a, double *d_a, int x, int y)
{
    // x: neuron, y: feature. m: max_neuron.
    unsigned id_x, id_y, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);

    if (id_x < x && id_y < y)
    {

        lin_idx = id_x + id_y * x;
        a[lin_idx] = 1.0f / (1 + exp(-1 * a[lin_idx]));
        d_a[lin_idx] = a[lin_idx] * (1 - a[lin_idx]);
    }
}

__global__ void gpu::matrixLinear(double *a, double *d_a, int x, int y)
{
    // x: neuron, y: batch.
    unsigned id_x, id_y, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);
    lin_idx = id_x + id_y * x;

    if (id_x < x && id_y < y)
    {
        d_a[lin_idx] = 1;
    }
}

__global__ void gpu::matrixSoftmax(double *a, double *softmax_sum, double *d_a, unsigned x, unsigned y)
{
    unsigned i, id_x, id_y, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);
    lin_idx = id_x + (id_y * x);

    // Softmax calculation.
    if (id_x < x && id_y < y)
        d_a[lin_idx] = a[lin_idx] = exp(a[lin_idx]);
    __syncthreads();

    if (!id_x)
    {
        if (id_y < y)
        {
            softmax_sum[id_y] = 0;
            for (i = lin_idx; i < lin_idx + x; i++)
                softmax_sum[id_y] += a[i];
        }
    }
    __syncthreads();

    if (id_x < x && id_y < y)
        a[lin_idx] = a[lin_idx] / softmax_sum[id_y];
    __syncthreads();

    // Softmax derivative calculation.
    d_a[lin_idx] = 1;
    // if (id_x < x && id_y < y)
    // {
    //     d_a[lin_idx] = 0.0;
    //     for (i = 0; i < x; i++)
    //         d_a[lin_idx] +=a[lin_idx] * ((i == id_x) - a[i + id_y * x]);
    // }
}

__global__ void gpu::matrixSquaredError(double *a, double *b, unsigned x, unsigned y)
{
    unsigned id_x, id_y, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);
    lin_idx = id_x + id_y * x;

    if (id_x < x && id_y < y)
        b[lin_idx] = pow((a[lin_idx]), 2);
}

__global__ void gpu::matrixSqrt(double *a, double *b, unsigned x, unsigned y)
{
    // input dimension xy
    // output dimension xy
    unsigned id_x, id_y, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);
    lin_idx = id_x + id_y * x;

    if (id_x < x && id_y < y)
        b[lin_idx] = sqrt(a[lin_idx]);
}

__global__ void gpu::matrixFindMean(double *a, unsigned x, unsigned y, unsigned mean)
{
    unsigned indx_x, indx_y, inp_lin;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;
    inp_lin = indx_x + indx_y * x;

    if (indx_x < x & indx_y < y)
        a[inp_lin] /= mean;
}

__global__ void gpu::matrixDifference(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y)
{
    unsigned id_x, id_y, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);

    lin_idx = id_x + id_y * x;

    if (id_x < x && id_y < y)
    {
        output_C[lin_idx] = input_A[lin_idx] - input_B[lin_idx]; // a[lin_idx] - b[id_y];
    }
}

__global__ void gpu::matrixCrossEntropy(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y, unsigned z)
{
    unsigned id_y, i;
    extern __shared__ double cost[];
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);

    cost[id_y] = 0;

    for (i = 0; i < x; i++)
    {
        cost[id_y] += input_B[i + id_y * x] * log(input_A[i + id_y * x]);
    }

    __syncthreads();

    if (!id_y)
    {
        *(output_C) = 0;
        for (i = 0; i < y; i++)
            *(output_C) += cost[i];

        *(output_C) /= y;
    }
}

__global__ void gpu::matrixCrossEntropyDifference(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y, unsigned z)
{
    unsigned id_x, id_y, id_z, lin_idx;
    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);
    id_z = threadIdx.z + (blockDim.z * blockIdx.z);

    lin_idx = id_x + id_y * x + id_z * x * y;

    // Calculating only difference

    if (id_x < x && id_y < y && id_z < z)
    {
        output_C[lin_idx] = input_A[lin_idx] - input_B[lin_idx];
    }
}

__global__ void gpu::matrixConfusionMatrix(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y)
{
    unsigned j, intr_x, intr_y, lin_idx_inpA, lin_idx_inpB, lin_idx_out;

    intr_x = threadIdx.x + (blockIdx.x * blockDim.x);
    intr_y = threadIdx.y + (blockIdx.y * blockDim.y);
    lin_idx_out = intr_x + intr_y * x;

    if (intr_x < x && intr_y < y)
    {
        output_C[lin_idx_out] = 0;
        for (j = 0; j < y; j++)
        {
            lin_idx_inpA = intr_x + j * x;
            lin_idx_inpB = intr_y + j * x;
            output_C[lin_idx_out] += (input_A[lin_idx_inpA] && input_B[lin_idx_inpB]);
        }
    }
}

__global__ void gpu::matrixBinaryCrossEntropy(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y)
{

    unsigned id_x, id_y, lin_idx;
    double numerator, denominator;

    id_x = threadIdx.x + (blockDim.x * blockIdx.x);
    id_y = threadIdx.y + (blockDim.y * blockIdx.y);

    lin_idx = id_x + id_y * x;

    // Calculating only difference or partial derivative of binary cross entropy

    if (id_x < x && id_y < y)
    {
        numerator = input_A[lin_idx] - input_B[lin_idx];
        denominator = input_A[lin_idx] * (1 - input_A[lin_idx]);
        output_C[lin_idx] = numerator / denominator;
    }
}

__global__ void gpu::matrixUpdateParameters(double *weights_biases, double *learning_rate, double *d_weights_biases, unsigned a_m, unsigned a_n, unsigned a_o)
{
    unsigned indx_x, indx_y, indx_z, index;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;
    indx_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (indx_x < a_m && indx_y < a_n && indx_z < a_o)
    {
        index = indx_x + indx_y * a_m + indx_z * a_m * a_n;
        weights_biases[index] -= learning_rate[index] * d_weights_biases[index]; //  ;
    }
}

/// @brief Calculates weighted sum of input ndarray and output ndarray
/// @tparam None
/// @param input 2d array
/// @param output 2d array
/// @param x size of x dimension
/// @param y size of y dimension
/// @param alpha is weighted sum parameter
/// @return output (call by referance)
__global__ void gpu::maritrxWeightedSum(double *input, double *output, unsigned x, unsigned y, double alpha)
{
    unsigned id_x, id_y, index;

    id_x = threadIdx.x + (blockIdx.x * blockDim.x);
    id_y = threadIdx.y + (blockIdx.y * blockDim.y);

    if (id_x < x && id_y < y)
    {
        index = id_x + id_y * x;
        output[index] = alpha * output[index] + (1 - alpha) * input[index];
    }
}

/// @brief Normalize data of 2d input on y axis
/// @tparam None
/// @param input 2d array
/// @param std_div 1d array in x dimension
/// @param mean 1d array in x dimension
/// @param output 2d array
/// @param x size of x dimension
/// @param y size of y dimension
/// @return output (call by referance)
__global__ void gpu::matrixNormalize(double *input, double *std_div, double *mean, double *output, unsigned x, unsigned y)
{
    unsigned id_x, id_y, lin_idx;
    double denominator;

    id_x = threadIdx.x + (blockIdx.x * blockDim.x);
    id_y = threadIdx.y + (blockIdx.y * blockDim.y);

    if (id_x < x && id_y < y)
    {
        lin_idx = id_x + id_y * x;

        denominator = std_div[id_x] ? std_div[id_x] : 0.0001;

        output[lin_idx] = (input[lin_idx] - mean[id_x]) / denominator;
    }
}

/// @brief Applies scalling on normalized data on y axis
/// @tparam None
/// @param input 2d array
/// @param gamma 1d array in x dimension, scaling factor
/// @param beta 1d array in x dimension, bias factor
/// @param output 2d array
/// @param x size of x dimension
/// @param y size of y dimension
/// @return output (call by referance)
__global__ void gpu::matrixNormalScaling(double *input, double *gamma, double *beta, double *output, unsigned x, unsigned y)
{
    unsigned id_x, id_y, lin_idx;

    id_x = threadIdx.x + (blockIdx.x * blockDim.x);
    id_y = threadIdx.y + (blockIdx.y * blockDim.y);

    if (id_x < x && id_y < y)
    {
        lin_idx = id_x + id_y * x;

        output[lin_idx] = input[lin_idx] * gamma[id_x] + beta[id_x];
    }
}

__global__ void gpu::matrixExponentiallyWeightedMovingAvg(double sigma, double *sum_d_weights_biases, double *d_weights_biases, unsigned a_m, unsigned a_n, unsigned a_o)
{
    unsigned indx_x, indx_y, indx_z, index;
    // double weighted_delta_weights_biases;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;
    indx_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (indx_x < a_m && indx_y < a_n && indx_z < a_o)
    {
        index = indx_x + indx_y * a_m + indx_z * a_m * a_n;

        d_weights_biases[index] = sum_d_weights_biases[index] = sigma * sum_d_weights_biases[index] + (1 - sigma) * d_weights_biases[index];
    }
}

__global__ void gpu::matrixUpdateWeightsBiasesRMSprop(double sigma, double epsalon, double *sum_d_weights_biases, double *d_weights_biases, unsigned a_m, unsigned a_n, unsigned a_o)
{
    unsigned indx_x, indx_y, indx_z, index;
    double squared_delta_weights_biases;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;
    indx_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (indx_x < a_m && indx_y < a_n && indx_z < a_o)
    {
        index = indx_x + indx_y * a_m + indx_z * a_m * a_n;

        squared_delta_weights_biases = pow(d_weights_biases[index], 2);
        sum_d_weights_biases[index] = sigma * sum_d_weights_biases[index] + (1 - sigma) * squared_delta_weights_biases;
        d_weights_biases[index] /= sqrt(sum_d_weights_biases[index] + epsalon);
    }
}

__global__ void gpu::matrixUpdateWeightsBiasesADAM(double *sigma, double *epsalon, double *learning_rate, double *sum_d_weights_biases, double *sum_d_weights_biases_squared, double *d_weights_biases, double *weights_biases, unsigned a_m, unsigned a_n)
{
    unsigned indx_x, indx_y, index;
    double squared_delta_weights_biases;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (indx_x < a_m && indx_y < a_n)
    {
        index = indx_x + indx_y * a_m;

        squared_delta_weights_biases = pow(d_weights_biases[index], 2);

        sum_d_weights_biases[index] = (*sigma) * sum_d_weights_biases[index] + (1 - (*sigma)) * d_weights_biases[index];
        sum_d_weights_biases_squared[index] = (*sigma) * sum_d_weights_biases_squared[index] + (1 - (*sigma)) * squared_delta_weights_biases;

        weights_biases[index] -= learning_rate[index] * sum_d_weights_biases[index] / (sqrt(sum_d_weights_biases_squared[index]) + (*epsalon)); //  ;
    }
}

__global__ void gpu::matrixUpdateLearningRateAdagrad(double epsalon, double learning_rate, double *learning_rate_eta, double *delta_weights_biases, double *sum_delta_weights, unsigned a_m, unsigned a_n, unsigned a_o)
{
    unsigned indx_x, indx_y, indx_z, index;
    double squared_delta_weights_biases;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;
    indx_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (indx_x < a_m && indx_y < a_n && indx_z < a_o)
    {
        index = indx_x + indx_y * a_m + indx_z * a_m * a_n;

        squared_delta_weights_biases = pow(delta_weights_biases[index], 2);
        sum_delta_weights[index] += squared_delta_weights_biases;

        learning_rate_eta[index] = learning_rate / sqrt(sum_delta_weights[index] + epsalon);
    }
}

__global__ void gpu::matrixUpdateLearningRateAdadelta(double epsalon, double sigma, double *delta_weights_biases, double *sum_delta_weights, double *learning_rate, double *eta_learning_rate, unsigned a_m, unsigned a_n, unsigned a_o)
{
    unsigned indx_x, indx_y, indx_z, index;
    double squared_delta_weights_biases;

    indx_x = threadIdx.x + blockIdx.x * blockDim.x;
    indx_y = threadIdx.y + blockIdx.y * blockDim.y;
    indx_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (indx_x < a_m && indx_y < a_n && indx_z < a_o)
    {
        index = indx_x + indx_y * a_m + indx_z * a_m * a_n;

        squared_delta_weights_biases = pow(delta_weights_biases[index], 2);
        sum_delta_weights[index] = sigma * sum_delta_weights[index] + (1 - sigma) * squared_delta_weights_biases;
        eta_learning_rate[index] = learning_rate[index] / sqrt(sum_delta_weights[index] + epsalon);
    }
}

/*
template <typename T>
void GPU_aux<T>::allocateGPUMemory(T *data, unsigned n)
{
    cudaMalloc((T **)&data, nElem * sizeof(T));
}

template <typename T>
void GPU_aux<T>::getCUDADeviceCount(unsigned *no_of_gpu)
{
    cudaGetDeviceCount(no_of_gpu);
}

template <typename T>
void GPU_aux<T>::printCUDAElement(T *data)
{
    gpu::print<<<1, 1>>>(data);
    cudaDeviceSynchronize();
}

template <typename T>
void GPU_aux<T>::cudaMemoryCopyHostToDevice(T *data_destination, T *data_soruce, unsigned nElem)
{
    cudaMemcpy(data_destination, data_source, sizeof(T) * nElem, cudaMemcpyHostToDevice);
}

template <typename T>
void GPU_aux<T>::cudaMemoryDeviceToDevice(T *data_destination, T *data_soruce, unsigned nElem)
{
    cudaMemcpy(data_destination, data_source, sizeof(T) * nElem, cudaMemcpyDeviceToDevice);
}

template <typename T>
void GPU_aux<T>::cudaMemoryCopyDeviceToHost(T *data_destination, T *data_soruce, unsigned nElem)
{
    cudaMemcpy(data_destination, data_source, sizeof(T) * nElem, cudaMemcpyHostToDevice);
}

template <typename T>
void GPU_aux<T>::cudaMemoryCopyDeviceToHost(T *data)
{
    cudaFree(data);
}
    */