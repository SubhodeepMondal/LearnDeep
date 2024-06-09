#include <map>
#include "GPULibrary.h"
#include "NDynamicArray.h"
#include "OptimizationType.h"
#include "Optimizers.h"

DenseOptimizer::DenseOptimizer(std::string optimization, NDArray<double, 1> *input_gpu, NDArray<double, 1> *nd_arrays)
{

    // Mapping vectors properly
    input = input_gpu;
    weights = nd_arrays[1];
    biases = nd_arrays[2];

    delta_activation = nd_arrays[3];
    difference = nd_arrays[4];
    delta_input = nd_arrays[5];

    axis_dims = new unsigned[3];

    axis_dims[0] = weights.getDimensions()[0]; // No of neurons
    axis_dims[1] = weights.getDimensions()[1]; // No of features/incoming input dimention
    axis_dims[2] = input->getDimensions()[1];  // Batch size

    delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
    delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());

    learning_rate_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
    learning_rate_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());

    delta_weights_intermediate = NDArray<double, 1>(weights.getNoOfDimensions() + 1, axis_dims);
    delta_biases_intermediate = NDArray<double, 1>(biases.getNoOfDimensions() + 1, axis_dims[0], axis_dims[2]);
    delta_input_intermediate = NDArray<double, 1>(delta_input.getNoOfDimensions() + 1, axis_dims[1], axis_dims[2], axis_dims[0]);

    learning_rate_weights.initData(learning_rate);
    learning_rate_biases.initData(learning_rate);

    nd_vectors_weights[0] = weights;
    nd_vectors_weights[1] = delta_weights;
    nd_vectors_weights[2] = learning_rate_weights;

    nd_vectors_biases[0] = biases;
    nd_vectors_biases[1] = delta_biases;
    nd_vectors_biases[2] = learning_rate_biases;

    // Assigning Optimizers Type for both weights and biases
    switch (optimizers.optimizer[optimization])
    {
    case this->optimizers.sgd:
    {
        this->optimization_type_weights = new SGD(nd_vectors_weights);
        this->optimization_type_biases = new SGD(nd_vectors_biases);
        break;
    }
    case this->optimizers.adagrad:
    {
        this->optimization_type_weights = new Adagrad(nd_vectors_weights);
        this->optimization_type_biases = new Adagrad(nd_vectors_biases);
        break;
    }
    case this->optimizers.adadelta:
    {
        this->optimization_type_weights = new Adadelta(nd_vectors_weights);
        this->optimization_type_biases = new Adadelta(nd_vectors_biases);
        break;
    }
    case this->optimizers.sgd_momentum:
    {
        this->optimization_type_weights = new SGD_momentum(nd_vectors_weights);
        this->optimization_type_biases = new SGD_momentum(nd_vectors_biases);
        break;
    }
    case this->optimizers.rmsprop:
    {
        this->optimization_type_weights = new RMSprop(nd_vectors_weights);
        this->optimization_type_biases = new RMSprop(nd_vectors_biases);
        break;
    }
    case this->optimizers.adam:
    {
        this->optimization_type_weights = new ADAM(nd_vectors_weights);
        this->optimization_type_biases = new ADAM(nd_vectors_biases);
        break;
    }
    default:
        this->optimization_type_weights = new SGD(nd_vectors_weights);
        this->optimization_type_biases = new SGD(nd_vectors_biases);
        break;
    }
}

/// @brief Overloaded method of Optimizer which optimizes based on optimization type
/// @tparam None
/// @param stream CudaStream for gpu stream calculation
/// @return output (call by referance)
void DenseOptimizer::optimize(cudaStream_t stream)
{

    // Calculating delta weights

    NDArray<double, 1> input_gpu = *input;
    block.x = (axis_dims[0] > 8) ? 8 : axis_dims[0];
    block.y = (axis_dims[1] > 8) ? 8 : axis_dims[1];
    block.z = (axis_dims[2] > 16) ? 16 : axis_dims[2];

    grid.x = ceil((float)axis_dims[0] / block.x);
    grid.y = ceil((float)axis_dims[1] / block.y);
    grid.z = ceil((float)axis_dims[2] / block.z);

    gpu::matrixDifferentialParameters<<<grid, block, 0, stream>>>(input_gpu.getData(), delta_activation.getData(), difference.getData(), delta_weights_intermediate.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);

    block.z = 1;
    grid.z = 1;
    gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_weights_intermediate.getData(), delta_weights.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
    gpu::matrixFindMean<<<grid, block, 0, stream>>>(delta_weights.getData(), axis_dims[0], axis_dims[1], axis_dims[2]);
    

    // Calculating delta biases
    gpu::matrixDifferentialParameters<<<grid, block, 0, stream>>>(NULL, delta_activation.getData(), difference.getData(), delta_biases_intermediate.getData(), axis_dims[0], axis_dims[2], 1);

    block.y = 1;
    grid.y = 1;
    gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_biases_intermediate.getData(), delta_biases.getData(), axis_dims[0], 1, axis_dims[2]);
    gpu::matrixFindMean<<<grid, block, 0, stream>>>(delta_biases.getData(), axis_dims[0], 1, axis_dims[2]);


    // Calculating backpropagation input/ difference
    block.x = (axis_dims[1] > 8) ? 8 : axis_dims[1];
    block.y = (axis_dims[2] > 8) ? 8 : axis_dims[2];
    block.z = (axis_dims[0] > 16) ? 16 : axis_dims[0];

    grid.x = ceil((float)axis_dims[1] / block.x);
    grid.y = ceil((float)axis_dims[2] / block.y);
    grid.z = ceil((float)axis_dims[0] / block.z);

    gpu::matrixDifferentialParameters<<<grid, block, 0, stream>>>(weights.getData(), delta_activation.getData(), difference.getData(), delta_input_intermediate.getData(), axis_dims[1], axis_dims[2], axis_dims[0]);
    
    block.z = 1;
    grid.z = 1;
    gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_input_intermediate.getData(), delta_input.getData(), axis_dims[1], axis_dims[2], axis_dims[0]);
    

    optimization_type_weights->optimize(stream);
    optimization_type_biases->optimize(stream);
}
