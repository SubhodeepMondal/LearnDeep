#pragma ONCE

class OptimizationType
{
public:
    virtual void optimize(cudaStream_t stream) = 0;
};

class SGD : public OptimizationType
{
    double learning_rate;
    NDArray<double, 1> parameters, delta_parameters, learning_rate_parameters;
    dim3 grid, block;
    unsigned *axis_dims;

public:
    SGD(NDArray<double, 1> *);

    void optimize(cudaStream_t);
};

class SGD_momentum : public OptimizationType
{
    unsigned *axis_dims;
    unsigned isFirstIter;
    double epsalon, sigma, learning_rate;
    NDArray<double, 1> parameters, delta_parameters, sum_delta_parameters, parameters_learning_rate;
    dim3 grid, block;

public:
    SGD_momentum(NDArray<double, 1> *);

    void optimize(cudaStream_t);
};

class Adagrad : public OptimizationType
{
    unsigned *axis_dims;
    double epsalon, learning_rate;
    NDArray<double, 1> parameters_learning_rate, eta_learning_rate;
    NDArray<double, 1> parameters, delta_parameters, sum_delta_parameters, delta_parameters_intermediate;
    dim3 block, grid;

public:
    Adagrad(NDArray<double, 1> *);

    void optimize(cudaStream_t);
};

class Adadelta : public OptimizationType
{
    unsigned *axis_dims;
    double epsalon, sigma, learning_rate;
    NDArray<double, 1> parameters, delta_parameters, sum_delta_parameters, delta_parameters_intermediate;
    NDArray<double, 1> parameters_learning_rate, eta_learning_rate;
    dim3 grid, block;

public:
    Adadelta(NDArray<double, 1> *);

    void optimize(cudaStream_t);
};

class RMSprop : public OptimizationType
{
    unsigned *axis_dims, isFirstIter;
    double epsalon, sigma, learning_rate;
    NDArray<double, 1> parameters, delta_parameters, sum_delta_parameters;
    NDArray<double, 1> parameters_learning_rate;
    dim3 grid, block;

public:
    RMSprop(NDArray<double, 1> *);

    void optimize( cudaStream_t);
};

class ADAM : public OptimizationType
{
    unsigned *axis_dims, isFirstIter;
    NDArray<double, 1> parameters, delta_parameters, sum_delta_parameters, squared_parameters_intermediate;
    NDArray<double, 1> parameters_learning_rate, eta_learning_rate;
    double epsalon, sigma, gamma, learning_rate;
    dim3 grid, block;

public:
    ADAM(NDArray<double, 1> *);

    void optimize(cudaStream_t);
};