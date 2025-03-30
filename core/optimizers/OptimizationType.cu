#include "GPULibrary.h"
#include "NDynamicArray.h"
#include "OptimizationType.h"

SGD::SGD(NDArray<double, 1> *nd_vectors)
{
    axis_dims = new unsigned[3];
    learning_rate = 0.01;

    parameters = nd_vectors[0];
    delta_parameters = nd_vectors[1];

    learning_rate_parameters = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());

    learning_rate_parameters.initData(learning_rate);
}

void SGD::optimize(cudaStream_t stream)
{
    // Calculating delta parameter

    for (unsigned i = 0; i < 3; i++)
        if (i < parameters.getNoOfDimensions())
            axis_dims[i] = parameters.getDimensions()[i];
        else
            axis_dims[i] = 1;

    // for(int i = 0; i< 3; i++)
    //     std::cout << axis_dims[i] << " ";
    // std::cout << "\n";
    block.x = (axis_dims[0] > 8) ? 8 : axis_dims[0];
    block.y = (axis_dims[1] > 8) ? 8 : axis_dims[1];
    block.z = (axis_dims[2] > 16) ? 16 : axis_dims[2];

    grid.x = ceil((float)axis_dims[0] / block.x);
    grid.y = ceil((float)axis_dims[1] / block.y);
    grid.z = ceil((float)axis_dims[2] / block.y);

    // std::cout << "parameter\n";
    // parameters.printData();
    // std::cout << "\n";

    // std::cout << "lr\n";
    // learning_rate_parameters.printData();
    // std::cout << "\n";
    // std::cout << "delta parameters\n";
    // delta_parameters.printData();
    // std::cout << "\n";
    gpu::matrixUpdateParameters<<<grid, block, 0, stream>>>(parameters.getData(), learning_rate_parameters.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
}

SGD_momentum::SGD_momentum(NDArray<double, 1> *nd_vectors)
{
    epsalon = 1.1025;
    sigma = 0.99;
    isFirstIter = 1;
    learning_rate = 0.01;

    axis_dims = new unsigned[3];
    parameters = nd_vectors[0];
    delta_parameters = nd_vectors[1];
    // parameters_learning_rate = nd_vectors[2];

    parameters_learning_rate = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());
    sum_delta_parameters = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());

    parameters_learning_rate.initData(learning_rate);
};

void SGD_momentum::optimize(cudaStream_t stream)
{

    for (unsigned i = 0; i < 3; i++)
        if (i < parameters.getNoOfDimensions())
            axis_dims[i] = parameters.getDimensions()[i];
        else
            axis_dims[i] = 1;

    block.x = (axis_dims[0] > 8) ? 8 : axis_dims[0];
    block.y = (axis_dims[1] > 8) ? 8 : axis_dims[1];
    block.z = (axis_dims[2] > 16) ? 16 : axis_dims[2];

    grid.x = ceil((float)axis_dims[0] / block.x);
    grid.y = ceil((float)axis_dims[1] / block.y);
    grid.z = ceil((float)axis_dims[2] / block.y);

    if (isFirstIter)
    {
        sum_delta_parameters.initData(parameters);
        isFirstIter = 0;
    }
    else
        gpu::matrixExponentiallyWeightedMovingAvg<<<grid, block, 0, stream>>>(sigma, sum_delta_parameters.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
    gpu::matrixUpdateParameters<<<grid, block, 0, stream>>>(parameters.getData(), parameters_learning_rate.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
}

Adagrad::Adagrad(NDArray<double, 1> *nd_vectors)
{
    epsalon = 1.1025;
    learning_rate = 0.1;
    axis_dims = new unsigned[3];

    parameters = nd_vectors[0];
    delta_parameters = nd_vectors[1];
    // parameters_learning_rate = nd_vectors[2];

    sum_delta_parameters = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());
    eta_learning_rate = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());
    parameters_learning_rate = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());

    // parameters_learning_rate.initData(learning_rate);
    eta_learning_rate.initData(parameters_learning_rate);
    sum_delta_parameters.initData((double)0.0);
}

void Adagrad::optimize(cudaStream_t stream)
{

    for (unsigned i = 0; i < 3; i++)
        if (i < parameters.getNoOfDimensions())
            axis_dims[i] = parameters.getDimensions()[i];
        else
            axis_dims[i] = 1;


    block.x = (axis_dims[0] > 8) ? 8 : axis_dims[0];
    block.y = (axis_dims[1] > 8) ? 8 : axis_dims[1];
    block.z = (axis_dims[2] > 16) ? 16 : axis_dims[2];

    grid.x = ceil((float)axis_dims[0] / block.x);
    grid.y = ceil((float)axis_dims[1] / block.y);
    grid.z = ceil((float)axis_dims[2] / block.y);

    gpu::matrixUpdateLearningRateAdagrad<<<grid, block, 0, stream>>>(epsalon, learning_rate, eta_learning_rate.getData(), delta_parameters.getData(), sum_delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);

    gpu::matrixUpdateParameters<<<grid, block, 0, stream>>>(parameters.getData(), eta_learning_rate.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
}

Adadelta::Adadelta(NDArray<double, 1> *nd_vectors)
{
    epsalon = 1.1025;
    sigma = 0.9;
    learning_rate = 0.01;

    axis_dims = new unsigned[3];
    parameters = nd_vectors[0];
    delta_parameters = nd_vectors[1];

    sum_delta_parameters = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());
    eta_learning_rate = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());
    parameters_learning_rate = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());

    parameters_learning_rate.initData(learning_rate);
    sum_delta_parameters.initData(parameters);
}

void Adadelta::optimize(cudaStream_t stream)
{

    for (unsigned i = 0; i < 3; i++)
        if (i < parameters.getNoOfDimensions())
            axis_dims[i] = parameters.getDimensions()[i];
        else
            axis_dims[i] = 1;

    block.x = (axis_dims[0] > 8) ? 8 : axis_dims[0];
    block.y = (axis_dims[1] > 8) ? 8 : axis_dims[1];
    block.z = (axis_dims[2] > 16) ? 16 : axis_dims[2];

    grid.x = ceil((float)axis_dims[0] / block.x);
    grid.y = ceil((float)axis_dims[1] / block.y);
    grid.z = ceil((float)axis_dims[2] / block.y);


    gpu::matrixUpdateLearningRateAdadelta<<<grid, block, 0, stream>>>(epsalon, sigma, delta_parameters.getData(), sum_delta_parameters.getData(), parameters_learning_rate.getData(), eta_learning_rate.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
    gpu::matrixUpdateParameters<<<grid, block, 0, stream>>>(parameters.getData(), eta_learning_rate.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
}

RMSprop::RMSprop(NDArray<double, 1> *nd_vectors)
{
    epsalon = 1.1025;
    sigma = 0.9;
    learning_rate = 0.05;

    axis_dims = new unsigned[3];
    parameters = nd_vectors[0];
    delta_parameters = nd_vectors[1];

    sum_delta_parameters = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());
    parameters_learning_rate = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());

    parameters_learning_rate.initData(learning_rate);
    sum_delta_parameters.initData(0.0);
}

void RMSprop::optimize(cudaStream_t stream)
{

    for (unsigned i = 0; i < 3; i++)
        if (i < parameters.getNoOfDimensions())
            axis_dims[i] = parameters.getDimensions()[i];
        else
            axis_dims[i] = 1;

    block.x = (axis_dims[0] > 8) ? 8 : axis_dims[0];
    block.y = (axis_dims[1] > 8) ? 8 : axis_dims[1];
    block.z = (axis_dims[2] > 16) ? 16 : axis_dims[2];

    grid.x = ceil((float)axis_dims[0] / block.x);
    grid.y = ceil((float)axis_dims[1] / block.y);
    grid.z = ceil((float)axis_dims[2] / block.y);

    gpu::matrixUpdateWeightsBiasesRMSprop<<<grid, block, 0, stream>>>(sigma, epsalon, sum_delta_parameters.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);

    gpu::matrixUpdateParameters<<<grid, block, 0, stream>>>(parameters.getData(), parameters_learning_rate.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
}

ADAM::ADAM(NDArray<double, 1> *nd_vectors)
{

    epsalon = 1.1025;
    sigma = 0.9;
    gamma = 0.95;
    learning_rate = 0.01;
    isFirstIter = 1;

    axis_dims = new unsigned[3];
    parameters = nd_vectors[0];
    delta_parameters = nd_vectors[1];
    parameters_learning_rate = nd_vectors[2];

    sum_delta_parameters = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());
    squared_parameters_intermediate = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());
    eta_learning_rate = NDArray<double, 1>(parameters.getNoOfDimensions(), parameters.getDimensions());

    parameters_learning_rate.initData(learning_rate);
    sum_delta_parameters.initData((double)0.0);
    squared_parameters_intermediate.initData((double)0.0);
}

void ADAM::optimize(cudaStream_t stream)
{

    for (unsigned i = 0; i < 3; i++)
        if (i < parameters.getNoOfDimensions())
            axis_dims[i] = parameters.getDimensions()[i];
        else
            axis_dims[i] = 1;

    // std::cout << "Parameters:\n";

    block.x = (axis_dims[0] > 8) ? 8 : axis_dims[0];
    block.y = (axis_dims[1] > 8) ? 8 : axis_dims[1];
    block.z = (axis_dims[2] > 16) ? 16 : axis_dims[2];

    grid.x = ceil((float)axis_dims[0] / block.x);
    grid.y = ceil((float)axis_dims[1] / block.y);
    grid.z = ceil((float)axis_dims[2] / block.y);

    gpu::matrixUpdateLearningRateAdadelta<<<grid, block, 0, stream>>>(epsalon, gamma, delta_parameters.getData(), squared_parameters_intermediate.getData(), parameters_learning_rate.getData(), eta_learning_rate.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);

    if (isFirstIter)
    {
        sum_delta_parameters.initData(parameters);
        isFirstIter = 0;
    }
    else
        gpu::matrixExponentiallyWeightedMovingAvg<<<grid, block, 0, stream>>>(sigma, sum_delta_parameters.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);

    gpu::matrixUpdateParameters<<<grid, block, 0, stream>>>(parameters.getData(), eta_learning_rate.getData(), delta_parameters.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
}
