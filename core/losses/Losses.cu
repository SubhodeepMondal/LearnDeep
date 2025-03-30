
#include <map>
#include "GPULibrary.h"
#include "NDynamicArray.h"
#include "MathLibrary.h"
#include "Losses.h"

/*Linear class*/
void Linear::findLoss(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream) {}

/*Mean_Squared_Error class*/
void Mean_Squared_Error::meanError(NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream)
{
    if (!isSquaredErrorAlocated)
    {
        squared_error = NDArray<double, 1>(Difference.getNoOfDimensions(), Difference.getDimensions());
        axis_dims = new unsigned[3];
        isSquaredErrorAlocated = 1;
    }

    math.squaredError(Difference, squared_error, stream);
    math.findMean(squared_error, Cost, stream);
}

void Mean_Squared_Error::findLoss(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream)
{
    math.findDifference(Y_predict, Y_target, Difference, stream);
    meanError(Difference, Cost, stream);

}

/*Catagorical_Cross_Entropy class*/
void Catagorical_Cross_Entropy::crossEntropyError(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Cost, NDArray<double, 1> Difference, cudaStream_t stream)
{
    if (isCrossEntropyErrorAlocated)
    {
        cross_entropy_error = NDArray<double, 1>(Difference.getNoOfDimensions(), Difference.getDimensions());
        axis_dims = new unsigned[3];
        isCrossEntropyErrorAlocated = 1;
    }
}

void Catagorical_Cross_Entropy::findLoss(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream)
{
    dim3 grid, block;

    for (int i = 0; i < 3; i++)
        if (i < Y_target.getNoOfDimensions())
            axis_dims[i] = Y_target.getDimensions()[i];
        else
            axis_dims[i] = 1;

    block.x = axis_dims[0] > 32 ? 32 : axis_dims[0];
    block.y = axis_dims[1] > 32 ? 32 : axis_dims[1];
    block.z = axis_dims[2] > 32 ? 32 : axis_dims[2];

    grid.x = ceil((float)axis_dims[0] / block.x);
    grid.y = ceil((float)axis_dims[1] / block.y);
    grid.z = ceil((float)axis_dims[2] / block.z);

    gpu::matrixCrossEntropyDifference<<<grid, block, 0, stream>>>(Y_predict.getData(), Y_target.getData(), Difference.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);


    block.x = 1;
    grid.x = 1;

    gpu::matrixCrossEntropy<<<grid, block, axis_dims[1] * sizeof(double), stream>>>(Y_predict.getData(), Y_target.getData(), Cost.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);

    cudaDeviceSynchronize();
}

/*Binary_Cross_Entropy class*/
void Binary_Cross_Entropy::findLoss(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream)
{
    math.binaryCrossEntropy(Y_predict, Y_target, Difference, Cost, stream);
}
