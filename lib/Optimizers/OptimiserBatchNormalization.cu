
#include <map>
#include "GPULibrary.h"
#include "NDynamicArray.h"
#include "OptimizationType.h"
#include "Optimizers.h"

OptimizerBatchNormalization::OptimizerBatchNormalization(std::string optimization, NDArray<double, 1> *nd_arrays)
{
    axis_dims = new unsigned[3];
    learning_rate_gamma = NDArray<double, 1>(gamma.getNoOfDimensions(), gamma.getDimensions());
    learning_rate_beta = NDArray<double, 1>(beta.getNoOfDimensions(), beta.getDimensions());
    delta_gamma = NDArray<double, 1>(gamma.getNoOfDimensions(), gamma.getDimensions());
    delta_beta = NDArray<double, 1>(beta.getNoOfDimensions(), beta.getDimensions());

    nd_vectors_gamma[0] = gamma;
    nd_vectors_gamma[1] = delta_gamma;
    nd_vectors_gamma[2] = learning_rate_gamma;
    nd_vectors_beta[0] = beta;
    nd_vectors_beta[1] = delta_beta;
    nd_vectors_beta[2] = learning_rate_beta;
    switch (optimizers.optimizer[optimization])
    {
    case this->optimizers.sgd:
    {
        this->optimization_type_gamma = new SGD(nd_vectors_gamma);
        this->optimization_type_beta = new SGD(nd_vectors_beta);

        // delete[] nd_vectors;
        break;
    }
    case this->optimizers.adagrad:
    {
        this->optimization_type_gamma = new Adagrad(nd_vectors_gamma);
        this->optimization_type_beta = new Adagrad(nd_vectors_beta);
        break;
    }
    case this->optimizers.adadelta:
    {
        this->optimization_type_gamma = new Adadelta(nd_vectors_gamma);
        this->optimization_type_beta = new Adadelta(nd_vectors_beta);
        break;
    }
    case this->optimizers.sgd_momentum:
    {
        this->optimization_type_gamma = new SGD_momentum(nd_vectors_gamma);
        this->optimization_type_beta = new SGD_momentum(nd_vectors_beta);
        break;
    }
    case this->optimizers.rmsprop:
    {
        this->optimization_type_gamma = new RMSprop(nd_vectors_gamma);
        this->optimization_type_beta = new RMSprop(nd_vectors_beta);
        break;
    }
    case this->optimizers.adam:
    {
        this->optimization_type_gamma = new ADAM(nd_vectors_gamma);
        this->optimization_type_beta = new ADAM(nd_vectors_beta);
        break;
    }
    default:
        break;
    }
    
}

void OptimizerBatchNormalization::optimize(cudaStream_t stream)
{
}