#pragma ONCE

typedef struct struct_Models
{
    enum Models
    {
        sequential,
        asequential,
        boltzman_machine
    };
    std::map<std::string, Models> model;
    struct_Models()
    {
        model["sequential"] = sequential;
        model["asequential"] = asequential;
        model["boltzman_machine"] = boltzman_machine;
    }
} Models;

class Model
{

protected:
    unsigned input_layer_count, output_layer_cout;
    NDArray<double, 1> X_input, y_target;
    NDArray<unsigned, 0> input_shape, output_shape;
    Layer *input, *output;
    std::string model_type;
    Optimizer *optimizer;
    Loss *loss;
    Metric *metric;
    struct_Models st_models;
    struct_Loss st_loss;
    struct_Metric st_metric;
    struct_Optimizer st_optimizer;
    search_parameter search;
    return_parameter ret_param;

public:
    Model(std::string str) ;

    virtual void add(Layer *layers) ;

    void getModelType();

    /// @brief Compiles the added layer to the model. allocates memory to the vectors. creates the graph for the model.
    /// @tparam None
    /// @param loss String value, describes which loss should be considered for optimization (i.e mae, )
    /// @param optimizer String value, describes which optimizer to be used
    /// @param metrics String value, which metrics to be used for model performance evaluation after each iteration.
    /// @return output (call by referance)
    virtual void compile(std::string loss, std::string optimizer, std::string metrics) ;

    /// @brief Sumarizes model with individual layer description
    /// @tparam None
    /// @return None
    void summary();

    /// @brief Fits the model with respective parameters and hyper parameters.
    /// @tparam None
    /// @param X Input in NDArray
    /// @param Y Target in NDArray
    /// @param epochs no of iterations training is going to repeat
    /// @param batch_size no of training instances for each epochs
    /// @return output (call by referance)
    void fit(NDArray<double, 0> X, NDArray<double, 0> Y, int epochs, int batch_size);


    /// @brief Fits the model with respective parameters and hyper parameters.
    /// @tparam None
    /// @param X Input in NDArray
    /// @param Y Target in NDArray
    /// @param epochs no of iterations training is going to repeat
    /// @param batch_size no of training instances for each epochs
    /// @return output (call by referance)
    NDArray<double, 0> predict(NDArray<double, 0> X);
};

class Sequential : public Model
{

public:
    Sequential() ;

    void add(Layer *layer);

    void compile(std::string loss, std::string optimizer, std::string metrics);
};