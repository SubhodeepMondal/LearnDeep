#pragma ONCE


typedef struct struct_Layers
{
    enum layers
    {
        dense,
        conv2d,
        batch_normalization,
        dropout,
    };
    std::map<std::string, layers> layer;

    struct_Layers()
    {
        layer["dense"] = dense;
        layer["conv2d"] = conv2d;
        layer["batch_normalization"] = batch_normalization;
        layer["dropout"] = dropout;
    }
} struct_Layers;

typedef struct search_parameter
{
    std::string search_param;
    std::string String;
    unsigned unsigned_index;
    double Double;
    int *int_Ptr;
    double *double_Ptr;
    cudaStream_t cuda_Stream;
    NDArray<double, 0> Data_CPU;
    NDArray<double, 1> Data_GPU;
} search_parameter;

typedef struct return_parameter
{
    std::string string_value;
    unsigned unsigned_index;
    double double_value;

    int *integer_ptr;
    double *double_ptr;

    cudaStream_t cuda_Stream;

    NDArray<double, 0> Data_CPU;
    NDArray<double, 1> Data_GPU;

} return_parameter;

typedef struct search_Flags
{
    enum flags
    {
        compile,
        train,
        predict,
        summary,
        prepare_training,
        initilize_weights_biases,
        initilize_weights_biases_gpu,
        forward_propagation,
        backward_propagation,
        initilize_optimizer,
        initilize_output,
        initilize_output_gpu,
        initilize_difference,
        print_output_gpu,
        update_stream,
        set_input_pointer,
        print_pointer,
        print_parameters,
        commit_weights_biases,
        initilize_input_intermidiate,
        initilize_output_intermidiate
    };

    std::map<std::string, flags> search_flags;

    search_Flags()
    {
        search_flags["compile"] = compile;
        search_flags["train"] = train;
        search_flags["predict"] = predict;
        search_flags["summary"] = summary;
        search_flags["prepare_training"] = prepare_training;
        search_flags["initilize_weights_biases"] = initilize_weights_biases;
        search_flags["initilize_weights_biases_gpu"] = initilize_weights_biases_gpu;
        search_flags["forward_propagation"] = forward_propagation;
        search_flags["backward_propagation"] = backward_propagation;
        search_flags["initilize_optimizer"] = initilize_optimizer;
        search_flags["initilize_output"] = initilize_output;
        search_flags["initilize_output_gpu"] = initilize_output_gpu;
        search_flags["initilize_input_intermidiate"] = initilize_input_intermidiate;
        search_flags["initilize_output_intermidiate"] = initilize_output_intermidiate;
        search_flags["initilize_difference"] = initilize_difference;
        search_flags["print_parameters"] = print_parameters;
        search_flags["update_stream"] = update_stream;
        search_flags["set_input_pointer"] = set_input_pointer;
        search_flags["print_pointer"] = print_pointer;
        search_flags["commit_weights_biases"] = commit_weights_biases;
    }
} search_Flags;

typedef struct search_Positions
{
    enum positions
    {
        input,
        output,
    };
    std::map<std::string, positions> search_positions;

    search_Positions()
    {
        search_positions["input"] = input;
        search_positions["output"] = output;
    }
} search_Positions;



/*Layer  Class*/
class Layer
{
protected:
    typedef struct Layer_ptr
    {
        struct Layer_ptr *next, *previous;
        Layer *layer;

        Layer_ptr()
        {
            next = previous = NULL;
            layer = NULL;
        }
    } Layer_ptr;

public:
    std::string layer_type;
    Layer_ptr *in_vertices, *out_vertices;

    Layer(std::string Layer_Type) : in_vertices(NULL), out_vertices(NULL), layer_type(Layer_Type) {}

    void operator=(Layer *layer);

    virtual NDArray<double, 0> getOutput() { return 0; }

    virtual NDArray<double, 1> getOutputDataGPU();

    virtual NDArray<double, 1> getDifferencefromPrevious() { return 0; }

    virtual NDArray<double, 1> getDifference() { return 0; }

    virtual unsigned getNoOfNeuron() { return 0; }

    virtual void initilizeInputGPU(double *ptr) {}

    virtual void initilizeTarget(double *ptr) {}

    virtual void LayerProperties() {}

    virtual return_parameter searchDFS(search_parameter search);

    virtual void searchBFS(search_parameter search) ;

    virtual void updateStream(cudaStream_t stream) {}

    virtual void updateOptimizer(std::string optimizer) {}

    virtual void printInput() {}

    virtual void printInputIntermediate() {}

    virtual void printTarget() {}

    virtual void printWeightes() {}

    virtual void printWeightGPU() {}

    virtual void printBiases() {}

    virtual void printBiasesGPU() {}

    virtual void printOutputGPU() {}

    virtual void printDifference() {}
};



/* Dense Layer Class*/

class Dense : public Layer, DenseForward
{
protected:
    Activation *activation;
    struct_Activations acti_func;
    std::string dense_activation; // dense activation function
    NDArray<double, 0> input;
    NDArray<double, 0> weights;
    NDArray<double, 0> biases;
    NDArray<double, 0> output;
    NDArray<double, 1> input_gpu;
    NDArray<double, 1> weights_gpu;
    NDArray<double, 1> biases_gpu;
    NDArray<double, 1> delta_activation;
    NDArray<double, 1> delta_weights;
    NDArray<double, 1> delta_biases;
    NDArray<double, 1> delta_prev_input;
    NDArray<double, 1> delta_input;
    NDArray<double, 1> output_gpu;
    NDArray<double, 1> difference;
    NDArray<double, 1> *nd_array_ptr = new NDArray<double, 1>[6];
    cudaStream_t stream;
    Optimizer *optimizer;
    struct_Optimizer optimizers;

    unsigned dense_unit, batch_size, isInputInitilized, isOptimizerUpdated = 0, isTrainable, isCUDAStreamUpdated = 0; // no of neuron of dense layer
    std::string layer_name, layer_optimizer;
    search_Flags Flag;

    NDArray<double, 1> getDifference();

    NDArray<double, 1> getDifferencefromPrevious();

    void initilizeInput();

    void initilizeInputIntermediate(unsigned batch_size);

    void initilizeWeightsBiases();

    void initilizeWeightsBiasesIntermediate(unsigned batch_size);

    void initilizeOutputIntermediate(unsigned batch_size);

    void initilizeActivation();

    void initilizeOptimizer(Optimizer *optimizer);

    void commitWeightsBiases();

public:
    Dense(unsigned , NDArray<double, 0> , std::string , std::string , unsigned isTrainable = 1  ) ;

    Dense(unsigned , std::string , std::string ) ;

    void layerProperties();

    unsigned getNoOfDimensions();

    unsigned *getDimensions();

    NDArray<double, 0> getOutput() override;

    NDArray<double, 1> getOutputDataGPU();

    std::string getActivationFunction();

    unsigned getNoOfNeuron();

    void updateStream(cudaStream_t stream);

    void updateOptimizer(std::string optimizer);

    return_parameter searchDFS(search_parameter search);

    void searchBFS(search_parameter search);

    void initilizeInputGPU(double *ptr) override ;

    void initilizeTarget(double *ptr) override ;

    void printInput();

    void printInputIntermediate();

    void printWeightes();

    void printWeightGPU();

    void printBiases();

    void printBiasesGPU();

    void printOutput();

    void printOutputGPU();

    void printDiffActivation();

    void printDifference();

    void printDiffWeightsBiases();
};




/* Batchnormalization Layer Class*/

class BatchNormalization : public Layer, BatchNormalizationForward
{
protected:
    std::string layer_name, layer_optimizer;
    NDArray<double, 0> input, beta, gamma, mean, std_div, output;
    NDArray<double, 1> beta_gpu, difference, gamma_gpu, input_gpu, mean_gpu, output_gpu, std_div_gpu, target;
    search_Flags Flag;
    unsigned isInputInitilized, isOptimizerUpdated = 0;
    cudaStream_t stream;
    Optimizer *optimizer;
    struct_Optimizer optimizers;

    void initilizeInput();

    void initilizeOutputIntermediate(unsigned batch_size);

    void initilizeInputIntermediate(unsigned batch_size);

    void initilizeBetaGammaIntermediate(unsigned batch_size);

    void initilizeBetaGamma();

    void initilizeOutput();

    void updateStream(cudaStream_t stream);

    void updateOptimizer(std::string optimizer);

public:
    BatchNormalization():Layer("batch normalization")
    {
        layer_name = "Batch_Normalization";
        isInputInitilized = 0;
    };
    BatchNormalization(std::string );

    BatchNormalization(NDArray<unsigned, 0> , std::string ) ;

    void layerProperties();

    NDArray<double, 0> getOutput() override ;

    return_parameter searchDFS(search_parameter search);

    void searchBFS(search_parameter search);

    void printGammaGPU();

    void printBetaGPU();
};