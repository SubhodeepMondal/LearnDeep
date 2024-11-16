#include <map>
#include "NDynamicArray.h"
#include "MathLibrary.h"
#include "Activations.h"
#include "OptimizationType.h"
#include "Optimizers.h"
#include "Forward_Propagation.h"
#include "Layers.h"

NDArray<double, 1> Dense::getDifference()
{
    return difference;
}

NDArray<double, 1> Dense::getDifferencefromPrevious()
{

    Layer_ptr *ptr = in_vertices;

    if (ptr)
    {
        return ptr->layer->getDifference();
    }
    else
        return 0;
}

void Dense::initilizeInput()
{
    Layer_ptr *ptr = in_vertices;
    NDArray<double, 0> input_dimension;
    if (!isInputInitilized)
    {
        if (ptr)
        {
            input_dimension = ptr->layer->getOutput();
            input = NDArray<double, 0>(input_dimension.getNoOfDimensions(), input_dimension.getDimensions(), 0);
            input.initPreinitilizedData(input_dimension.getData());
            isInputInitilized = 1;
        }
    }
}

void Dense::initilizeInputIntermediate(unsigned batch_size)
{
    unsigned dims = input.getNoOfDimensions() + 1;
    unsigned i;
    unsigned *a = new unsigned[dims];

    Layer_ptr *ptr = in_vertices;

    for (i = 0; i < dims - 1; i++)
        a[i] = input.getDimensions()[i];
    a[i] = batch_size;

    input_gpu = NDArray<double, 1>(dims, a, 0);

    std::cout << "input ptr" << input_gpu.getData();
    if (ptr)
    {
        input_gpu.initPreinitilizedData(ptr->layer->getOutputDataGPU().getData());
    }

    delete[] a;
}

void Dense::initilizeWeightsBiases()
{
    weights = NDArray<double, 0>(2, dense_unit, input.getDimensions()[0]);
    biases = NDArray<double, 0>(1, dense_unit);
    weights.initRandData(-1, 1);
    biases.initRandData(-1, 1);
}

void Dense::initilizeWeightsBiasesIntermediate(unsigned batch_size)
{
    unsigned no_of_features;
    no_of_features = input.getDimensions()[0];
    weights_gpu = NDArray<double, 1>(2, dense_unit, no_of_features);
    biases_gpu = NDArray<double, 1>(1, dense_unit);

    weights_gpu.initData(weights.getData());
    biases_gpu.initData(biases.getData());

    delta_input = getDifferencefromPrevious();
}

void Dense::initilizeOutputIntermediate(unsigned batch_size)
{
    unsigned no_of_dims, i;
    unsigned *a;

    no_of_dims = output.getNoOfDimensions() + 1;
    a = new unsigned[no_of_dims];

    for (i = 0; i < no_of_dims - 1; i++)
        a[i] = output.getDimensions()[i];
    a[i] = batch_size;

    output_gpu = NDArray<double, 1>(no_of_dims, a);
    difference = NDArray<double, 1>(no_of_dims, a);
    delta_activation = NDArray<double, 1>(no_of_dims, a);

    delete[] a;
}

void Dense::initilizeActivation()
{
    switch (acti_func.activations[dense_activation])
    {
    case this->acti_func.relu:
    {

        activation = new Relu_Activation;
        break;
    }
    case this->acti_func.sigmoid:
    {
        activation = new Sigmoid_Activation;
        break;
    }
    case this->acti_func.linear:
    {
        activation = new Linear_Activation;
        break;
    }
    case this->acti_func.softmax:
    {
        activation = new Softmax_Activation;
        break;
    }
    default:
    {
        activation = NULL;
        break;
    }
    }
}

void Dense::initilizeOptimizer(Optimizer *optimizer)
{
    this->optimizer = optimizer;
}

void Dense::commitWeightsBiases()
{
    weights.initData(weights_gpu);
    biases.initData(biases_gpu);
}

Dense::Dense(unsigned unit, NDArray<double, 0> input_shape, std::string activation, std::string layer_name = "dense", unsigned isTrainable) : Layer("dense")
{
    this->dense_unit = unit;
    this->dense_activation = activation;
    this->input = NDArray<double, 0>(input_shape.getNoOfDimensions(), input_shape.getDimensions()[0]);
    this->output = NDArray<double, 0>(1, unit);
    this->isInputInitilized = 1;
    this->layer_name = layer_name;
    this->isTrainable = isTrainable;
    initilizeActivation();
}

Dense::Dense(unsigned unit, std::string activation, std::string layer_name = "dense") : Layer("dense")
{
    this->dense_unit = unit;
    this->output = NDArray<double, 0>(1, unit);
    this->dense_activation = activation;
    this->isInputInitilized = 0;
    this->layer_name = layer_name;
    initilizeActivation();
}

void Dense::layerProperties()
{
    std::cout << layer_type << ": " << layer_name << " ";
    input.printDimensions();
    output.printDimensions();
    std::cout << " " << getActivationFunction() << "\n";
}

unsigned Dense::getNoOfDimensions()
{
    return input.getNoOfDimensions();
}

unsigned *Dense::getDimensions()
{
    return input.getDimensions();
}

NDArray<double, 0> Dense::getOutput()
{

    return output;
}

NDArray<double, 1> Dense::getOutputDataGPU()
{
    return output_gpu;
}

std::string Dense::getActivationFunction()
{
    return dense_activation;
}

unsigned Dense::getNoOfNeuron()
{
    return dense_unit;
}

void Dense::updateStream(cudaStream_t stream)
{
    if (!isCUDAStreamUpdated)
    {
        this->stream = stream;
        isCUDAStreamUpdated = 1;
        // std::cout << "updated stream:" << stream << "\n";
    }
}

void Dense::updateOptimizer(std::string optimizer)
{

    nd_array_ptr[0] = input_gpu;
    nd_array_ptr[1] = weights_gpu;
    nd_array_ptr[2] = biases_gpu;
    nd_array_ptr[3] = delta_activation;
    nd_array_ptr[4] = difference;
    nd_array_ptr[5] = delta_input;
    this->optimizer = new DenseOptimizer(optimizer, &input_gpu, nd_array_ptr);

    delete[] nd_array_ptr;
}

return_parameter Dense::searchDFS(search_parameter search)
{
    return_parameter return_param;
    Layer_ptr *ptr = out_vertices;

    switch (Flag.search_flags[search.search_param])
    {
    case this->Flag.summary:
    {
        layerProperties();
        break;
    }

    case this->Flag.compile:
    {
        initilizeInput();
        initilizeWeightsBiases();
        // initilizeOutput();
        layer_optimizer = search.String;
        break;
    }

    case this->Flag.prepare_training:
    {
        updateStream(search.cuda_Stream);
        initilizeOutputIntermediate(search.unsigned_index);
        initilizeInputIntermediate(search.unsigned_index);
        initilizeWeightsBiasesIntermediate(search.unsigned_index);
        updateOptimizer(layer_optimizer);

        break;
    }

    case this->Flag.initilize_input_intermidiate:
    {
        initilizeInputGPU(search.double_Ptr);
        return return_param;
        break;
    }

    case this->Flag.initilize_output_intermidiate:
    {
        initilizeTarget(search.double_Ptr);
        break;
    }

    case this->Flag.forward_propagation:
    {
        fit(input_gpu, weights_gpu, biases_gpu, output_gpu, stream);
        activation->activate(output_gpu, delta_activation, stream);
        if (!ptr)
        {
            return_param.Data_GPU = output_gpu;
            return return_param;
        }
        break;
    }

    case this->Flag.initilize_difference:
    {
        difference.initData(search.Data_GPU.getData());
        break;
    }

    case this->Flag.print_parameters:
    {
        printWeightGPU();
        printBiasesGPU();
        printWeightes();
        printBiases();
        break;
    }

    case this->Flag.predict:
    {
        predict(input, weights, biases, output);
        activation->activate(output);
        break;
    }

    case this->Flag.commit_weights_biases:
    {
        commitWeightsBiases();
        break;
    }
    }

    while (ptr)
    {
        return_param = ptr->layer->searchDFS(search);
        ptr = ptr->next;
    }
    return return_param;
}

void Dense::searchBFS(search_parameter search)
{
    Layer_ptr *ptr = in_vertices;

    switch (Flag.search_flags[search.search_param])
    {
    case this->Flag.compile:
    {
        initilizeInput();
        break;
    }
    case this->Flag.backward_propagation:
    {

        optimizer->optimize(stream);
        break;
    }
    }

    while (ptr)
        ptr = ptr->next;

    ptr = in_vertices;

    while (ptr)
    {

        ptr->layer->searchBFS(search);
        ptr = ptr->next;
    }
}

void Dense::initilizeInputGPU(double *ptr)
{
    input_gpu.initPreinitilizedData(ptr);
}

void Dense::initilizeTarget(double *ptr)
{
    // target.initPreinitilizedData(ptr);
}

void Dense::printInput()
{
    std::cout << layer_name << ": input data: \n";
    input.printData();
}

void Dense::printInputIntermediate()
{
    std::cout << layer_name << ": Input intermediate: \n";
    input_gpu.printData();
}

void Dense::printWeightes()
{
    std::cout << layer_name << ": weight data: \n";
    weights.printData();
}

void Dense::printWeightGPU()
{
    std::cout << layer_name << ": weight gpu data: \n";
    weights_gpu.printData();
}

void Dense::printBiases()
{
    std::cout << layer_name << ": biases data: \n";
    biases.printData();
}

void Dense::printBiasesGPU()
{
    std::cout << layer_name << ": biases gpu data: \n";
    biases_gpu.printData();
}

void Dense::printOutput()
{
    output.printData();
}

void Dense::printOutputGPU()
{
    std::cout << layer_name << ": output data: \n";
    output_gpu.printData();
}

void Dense::printDiffActivation()
{
    std::cout << layer_name << ": differential_activation: \n";
    delta_activation.printData();
}

void Dense::printDifference()
{
    std::cout << layer_name << ": Difference: \n";
    difference.printData();
}

void Dense::printDiffWeightsBiases()
{
    std::cout << layer_name << ": Differential Weights Biases: \n";
    delta_weights.printData();
}
