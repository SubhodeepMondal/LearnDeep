#include "NDynamicArray.h"
#include "MathLibrary.h"
#include "Forward_Propagation.h"


void BatchNormalizationForward::findMean(NDArray<double, 1> input_gpu, cudaStream_t stream)
{

    unsigned intr_x;
    dim3 grid, block;
    intr_x = input_gpu.getDimensions()[0];

    block.x = (intr_x > 8) ? 8 : intr_x;

    grid.x = ceil((float)intr_x / block.x);

    gpu::matrixRollingSum<<<grid, block, 0, stream>>>(input_gpu.getData(), mean.getData(), input_gpu.getDimensions()[0], 1, input_gpu.getDimensions()[1]);
    gpu::matrixFindMean<<<grid, block, 0, stream>>>(mean.getData(), mean.getDimensions()[0], 1, input_gpu.getDimensions()[1]);
}

void BatchNormalizationForward::findStdDiv(NDArray<double, 1> input_gpu, cudaStream_t stream)
{
    unsigned intr_x, intr_y;
    dim3 grid, block;
    intr_x = input_gpu.getDimensions()[0];
    intr_y = input_gpu.getDimensions()[1];

    block.x = (intr_x > 8) ? 8 : intr_x;
    block.y = (intr_y > 8) ? 8 : intr_y;

    grid.x = ceil((float)intr_x / block.x);
    grid.y = ceil((float)intr_y / block.y);

    gpu::matrixDifference<<<grid, block, 0, stream>>>(input_gpu.getData(), mean.getData(), stdDivTemp.getData(), input_gpu.getDimensions()[0], input_gpu.getDimensions()[1]);
    gpu::matrixSquaredError<<<grid, block, 0, stream>>>(stdDivTemp.getData(), stdDivTemp.getData(), input_gpu.getDimensions()[0], input_gpu.getDimensions()[1]);
    gpu::matrixRollingSum<<<grid, block, 0, stream>>>(stdDivTemp.getData(), stdDiv.getData(), input_gpu.getDimensions()[0], 1, input_gpu.getDimensions()[1]);

    block.y = 1;
    grid.y = 1;

    gpu::matrixFindMean<<<grid, block, 0, stream>>>(stdDiv.getData(), input_gpu.getDimensions()[0], 1, input_gpu.getDimensions()[1]);
    gpu::matrixSqrt<<<grid, block, 0, stream>>>(stdDiv.getData(), stdDiv.getData(), input_gpu.getDimensions()[0], input_gpu.getDimensions()[1]);
}

void BatchNormalizationForward::movingMean(cudaStream_t stream)
{
    unsigned intr_x;
    dim3 grid, block;

    double alpha = 0.9;

    intr_x = mean.getDimensions()[0];

    block.x = (intr_x > 8) ? 8 : intr_x;

    grid.x = ceil((float)intr_x / block.x);

    gpu::maritrxWeightedSum<<<grid, block, 0, stream>>>(mean.getData(), moving_mean.getData(), intr_x, 1, alpha);
}

void BatchNormalizationForward::movingStdDiv(cudaStream_t stream)
{

    unsigned intr_x;
    dim3 grid, block;
    double alpha = 0.9;

    intr_x = stdDiv.getDimensions()[0];

    block.x = (intr_x > 8) ? 8 : intr_x;

    grid.x = ceil((float)intr_x / block.x);

    gpu::maritrxWeightedSum<<<grid, block, 0, stream>>>(stdDiv.getData(), moving_stdDiv.getData(), intr_x, 1, alpha);
}

void BatchNormalizationForward::normalize(NDArray<double, 1> input_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream)
{

    unsigned intr_x, intr_y;
    dim3 grid, block;

    intr_x = input_gpu.getDimensions()[0];
    intr_y = input_gpu.getDimensions()[1];

    block.x = (intr_x > 8) ? 8 : intr_x;
    block.y = (intr_y > 8) ? 8 : intr_y;

    grid.x = ceil((float)intr_x / block.x);
    grid.y = ceil((float)intr_y / block.y);

    gpu::matrixNormalize<<<grid, block, 0, stream>>>(input_gpu.getData(), moving_stdDiv.getData(), moving_mean.getData(), output_gpu.getData(), intr_x, intr_y);
}

void BatchNormalizationForward::scale(NDArray<double, 1> gamma_gpu, NDArray<double, 1> beta_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream)
{
    unsigned intr_x, intr_y;
    dim3 grid, block;

    intr_x = output_gpu.getDimensions()[0];
    intr_y = output_gpu.getDimensions()[1];

    block.x = (intr_x > 8) ? 8 : intr_x;
    block.y = (intr_y > 8) ? 8 : intr_y;

    grid.x = ceil((float)intr_x / block.x);
    grid.y = ceil((float)intr_y / block.y);

    gpu::matrixNormalScaling<<<grid, block, 0, stream>>>(output_gpu.getData(), gamma_gpu.getData(), beta_gpu.getData(), output_gpu.getData(), intr_x, intr_y);
}

void BatchNormalizationForward::predict(NDArray<double, 0> input, NDArray<double, 0> gamma, NDArray<double, 0> beta, NDArray<double, 0> output)
{
}

void BatchNormalizationForward::fit(NDArray<double, 1> input_gpu, NDArray<double, 1> gamma_gpu, NDArray<double, 1> beta_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream)
{
    if (!isMeanStdDivInitilized)
    {
        unsigned no_of_features = input_gpu.getNoOfDimensions() - 1;
        unsigned *a = new unsigned[no_of_features];

        for (int i = 0; i < no_of_features; i++)
            a[i] = input_gpu.getDimensions()[i];

        mean = NDArray<double, 1>(no_of_features, a);
        moving_mean = NDArray<double, 1>(no_of_features, a);
        stdDiv = NDArray<double, 1>(no_of_features, a);
        moving_stdDiv = NDArray<double, 1>(no_of_features, a);
        stdDivTemp = NDArray<double, 1>(input_gpu.getNoOfDimensions(), input_gpu.getDimensions());

        // moving_mean.initRandData(0, 0);
        // moving_stdDiv.initRandData(0, 0);

        isMeanStdDivInitilized = 1;
        delete[] a;
    }

    findMean(input_gpu, stream);
    findStdDiv(input_gpu, stream);
    movingMean(stream);
    movingStdDiv(stream);
    normalize(input_gpu, output_gpu, stream);
    scale(gamma_gpu, beta_gpu, output_gpu, stream);

    std::cout << "input_data:\n";
    input_gpu.printData();
    std::cout << "std_data:\n";
    stdDiv.printData();
    std::cout << "mean_data:\n";
    mean.printData();
    std::cout << "moving_std_data:\n";
    moving_stdDiv.printData();
    std::cout << "moving_mean_data:\n";
    moving_mean.printData();
    std::cout << "gamma_data:\n";
    gamma_gpu.printData();
    std::cout << "beta_data:\n";
    beta_gpu.printData();
    std::cout << "output_data:\n";
    output_gpu.printData();
}