#pragma ONCE


typedef struct struct_Optimizer
{
    enum optimizers
    {
        sgd,
        sgd_momentum,
        adadelta,
        adafactor,
        adagrad,
        adam,
        adamw,
        ftrl,
        nadam,
        rmsprop
    };
    std::map<std::string, optimizers> optimizer;

    struct_Optimizer()
    {
        optimizer["sgd"] = sgd;
        optimizer["sgd_momentum"] = sgd_momentum;
        optimizer["adagrad"] = adagrad;
        optimizer["adadelta"] = adadelta;
        optimizer["adam"] = adam;
        optimizer["rmsprop"] = rmsprop;
        optimizer["nadam"] = nadam;
    }

} struct_Optimizer;




class Optimizer
{
protected:
    /* data */
    double learning_rate = 0.01;

    struct_Optimizer optimizers;

public:
    virtual void optimize(cudaStream_t stream) = 0;
};


class DenseOptimizer : public Optimizer
{

    unsigned *axis_dims;
    unsigned neurons, features, batches;
    NDArray<double, 1>* input;
    NDArray<double, 1> delta_weights;
    NDArray<double, 1> delta_biases;
    NDArray<double, 1> weights, biases, delta_activation, delta_input, difference;
    NDArray<double, 1> learning_rate_weights, learning_rate_biases, delta_weights_intermediate, delta_biases_intermediate, delta_input_intermediate;
    NDArray<double, 1> *nd_vectors_weights = new NDArray<double, 1> [3];
    NDArray<double, 1> *nd_vectors_biases = new NDArray<double, 1> [3];

    OptimizationType* optimization_type_weights;
    OptimizationType* optimization_type_biases ;

    dim3 grid, block;

public:
    DenseOptimizer(std::string ,NDArray<double, 1>*, NDArray<double, 1>*);

    /// @brief Overloaded method of Optimizer which optimizes based on optimization type
    /// @tparam None
    /// @param stream CudaStream for gpu stream calculation
    /// @return output (call by referance)
    void optimize(cudaStream_t stream);
};



class OptimizerBatchNormalization : public Optimizer
{

    unsigned *axis_dims;
    NDArray<double, 1> learning_rate_gamma;
    NDArray<double, 1> learning_rate_beta;
    NDArray<double, 1> gamma, delta_gamma, delta_gamma_intermediate;

    NDArray<double, 1> beta, delta_beta, delta_beta_intermediate;
    NDArray<double, 1> gamma_intermediate;
    NDArray<double, 1> beta_intermediate;

    NDArray<double, 1> *nd_vectors_gamma = new NDArray<double, 1> [3];
    NDArray<double, 1> *nd_vectors_beta = new NDArray<double, 1> [3];


    OptimizationType* optimization_type_gamma;
    OptimizationType* optimization_type_beta ;

public:
    // OptimizerBatchNormalization() {}
    OptimizerBatchNormalization(std::string, NDArray<double, 1>*);

    void optimize(cudaStream_t stream) override ;
};